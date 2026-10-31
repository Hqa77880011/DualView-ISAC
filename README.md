# DualView-ISAC

Implementation of **DualView-ISAC: Cross-View Self-Supervised Signal Representation Learning from Communication CSI and Radar Echoes**. Independent RF front ends learn shared motion embeddings through cross-view contrastive learning, bounded local temporal matching and full-to-single-view consistency.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Validation (2026-11-01): Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, PyTorch 2.7.1+cu118, scikit-learn 1.6.1; CPU, one PyTorch thread, seed 42. Actual paired complex CSI/radar NPZ records passed manifest loading, train-only normalization, token preprocessing, one epoch (dimension 8, batch 2), checkpoint loading and evaluation with a half-token timing offset and 50% missing views. Two tests passed: finite joint loss, nonzero gradients in both encoders/GRUs/fusion/heads, parameter update, reload agreement (rtol 1e-6, atol 1e-7), fractional shifts and hand-computed contrastive/F1/balanced-accuracy values. The four generated records check functionality and do not establish paper recognition performance.

## Data and preprocessing

Export synchronized recording intervals to NumPy archives with complex `csi[T,antenna,subcarrier]`, complex `radar[frame,chirp,fast_sample]`, and scalar `csi_fs`/`radar_fs` in samples or frames per second. Both arrays must refer to the same recording origin. The paired collection described by the manuscript is provided by its data owner; this interface also accepts paired measurements from your own acquisition system.

Create a manifest with paths relative to its location:

```json
[
  {"file": "raw/subject01_session01.npz", "group": "subject01", "room": "room1", "device": "link1", "split": "train", "label": 2},
  {"file": "raw/subject02_session01.npz", "group": "subject02", "room": "room1", "device": "link1", "split": "dev", "label": 0},
  {"file": "raw/subject03_session01.npz", "group": "subject03", "room": "room2", "device": "link2", "split": "test", "label": 1},
  {"file": "raw/subject04_session01.npz", "group": "subject04", "room": "room4", "device": "link2", "split": "transfer", "label": 5}
]
```

Label sit, stand, walk, lie down, pick up and wave as `0..5`; use `-1` for unlabeled training recordings. Each row describes one constant-activity interval. Group by the held-out subject, session or device identity appropriate to the experiment. Related recordings must remain in one partition; assign the held-out room to `transfer` before preparation.



Preparation creates 2.56-second windows with 50% overlap only in training and no overlap in later partitions. Each window becomes 16 temporal tokens. CSI channels contain mean amplitude, detrended adjacent-antenna phase difference, antenna correlation and local Doppler FFT magnitude. Radar channels contain mean range–Doppler magnitude and a range-summed micro-Doppler map. Maps are resized to 32×32. Channel normalization uses only training-window statistics, saved in `normalization.npz`.

```bash
dualview_isac prepare --manifest data/manifest.json --output data/prepared
```

## Training and inference

```bash
dualview_isac train --data data/prepared --output runs/train --seed 42
```

## Evaluation

```bash
dualview_isac evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

## Comparisons and ablations

```bash
dualview_isac train --data data/prepared --seed 42 --label-ratio 0.05 --output runs/labels05
dualview_isac train --data data/prepared --seed 42 --no-alignment --output runs/no_alignment
dualview_isac train --data data/prepared --seed 42 --no-viewdrop --output runs/no_viewdrop
dualview_isac train --data data/prepared --seed 42 --no-alignment --no-viewdrop --output runs/xview
dualview_isac train --data data/prepared --seed 42 --supervised-only --view csi --output runs/csi_supervised
dualview_isac train --data data/prepared --seed 42 --supervised-only --view radar --output runs/radar_supervised
dualview_isac train --data data/prepared --seed 42 --supervised-only --view full --output runs/late_fusion
dualview_isac evaluate --data data/prepared --checkpoint runs/train/model.pt --view csi --output runs/csi_path
dualview_isac evaluate --data data/prepared --checkpoint runs/train/model.pt --view radar --output runs/radar_path
dualview_isac evaluate --data data/prepared --checkpoint runs/train/model.pt --missing-ratio 0.5 --output runs/missing50
dualview_isac evaluate --data data/prepared --checkpoint runs/train/model.pt --offset 2 --output runs/offset2
dualview_isac evaluate --data data/prepared --checkpoint runs/train/model.pt --split transfer --output runs/transfer
```

Evaluate each fitted control with its own checkpoint; scoring/stress controls above use the full checkpoint.

## Outputs

Training writes `runs/train/model.pt`. Evaluation writes `metrics.json`, `predictions.npz`, `confusion.csv` and `confusion.png`. Prepared normalization is stored in `data/prepared/normalization.npz`.

## Method mapping and explicit settings

| Paper component | Code | Command |
| --- | --- | --- |
| CSI/radar preprocessing and train-only statistics | `dualview_isac/cli.py:prepare` | `dualview_isac prepare` |
| Independent CNN/GRU encoders and view heads | `dualview_isac/method.py` | `dualview_isac train` |
| Symmetric cross-view and bounded temporal contrast | `contrastive`, `temporal_contrastive` | `dualview_isac train --radius 2 --temperature 0.1 --temporal-weight 0.5` |
| Stop-gradient full-to-single view consistency | `dualview_isac/cli.py:train` | `dualview_isac train --view-weight 1` |
| Macro-F1, balanced accuracy, group scores, parameters, latency | `dualview_isac/cli.py:evaluate` | `dualview_isac evaluate` |
| Missing-view and offset controls | `shifted`, `evaluate` | `dualview_isac evaluate --missing-ratio 0.6 --offset 0.5` |

With the default 2.56-second/16-token preprocessing, one token is 160 ms and `--offset 0.5` is 80 ms. Fractional shifts linearly interpolate tokens with zero fill at boundaries. This is a token-domain timing stress, not raw RF resampling. Training applies a uniformly sampled integer radar shift with `--time-jitter 1`; `--time-jitter 0` disables it. Alignment radius is measured in tokens. The implementation averages coordinatewise squared consistency errors. The paper does not provide complete optimizer/objective settings; AdamW, learning rate 1e-3, 50 epochs, batch 32, embedding 128 and the displayed loss weights are explicit runnable choices. This CNN/GRU has its measured parameter count; it does not claim the paper's reported 0.82M-parameter branch architecture.

`--label-ratio 0` now retains no activity labels. For a positive ratio each represented training class retains at least one label. Groups must be partitioned before preparation. The supplied controls include CSI-only, radar-only and late fusion supervision; an early-fusion architecture is not implemented. The owned paired acquisition data and original baseline weights are not included, and no manuscript table was regenerated.

```bash
python -m unittest discover -s tests -v
```
