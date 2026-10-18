# DualView-ISAC

Implementation of **DualView-ISAC: Cross-View Self-Supervised Signal Representation Learning from Communication CSI and Radar Echoes**. Independent RF front ends learn shared motion embeddings through cross-view contrastive learning, bounded local temporal matching and full-to-single-view consistency.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

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
