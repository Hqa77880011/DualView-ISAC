# DualView-ISAC

Implementation of **DualView-ISAC: Cross-View Self-Supervised Signal Representation Learning from Communication CSI and Radar Echoes**. Independent RF front ends learn shared motion embeddings through cross-view contrastive learning, bounded local temporal matching and full-to-single-view consistency.

## Install and prepare recordings

Use Python 3.10+ and activate a virtual environment:

```bash
python -m venv .venv
python -m pip install -e .
```

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

```bash
dualview_isac prepare --manifest data/manifest.json --output data/prepared
dualview_isac train --data data/prepared --output runs/train
dualview_isac evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

Preparation creates 2.56-second windows with 50% overlap only in training and no overlap in later partitions. Each window becomes 16 temporal tokens. CSI channels contain mean amplitude, detrended adjacent-antenna phase difference, antenna correlation and local Doppler FFT magnitude. Radar channels contain mean range–Doppler magnitude and a range-summed micro-Doppler map. Maps are resized to 32×32. Channel normalization uses only training-window statistics, saved in `normalization.npz`.

## Controls and stress conditions

```bash
dualview_isac train --label-ratio 0.05 --output runs/labels05
dualview_isac train --no-alignment --output runs/no_alignment
dualview_isac train --no-viewdrop --output runs/no_viewdrop
dualview_isac train --no-alignment --no-viewdrop --output runs/xview
dualview_isac train --supervised-only --view csi --output runs/csi_supervised
dualview_isac train --supervised-only --view radar --output runs/radar_supervised
dualview_isac train --supervised-only --view full --output runs/late_fusion
dualview_isac evaluate --view csi --output runs/csi_path
dualview_isac evaluate --view radar --output runs/radar_path
dualview_isac evaluate --missing-ratio 0.5 --output runs/missing50
dualview_isac evaluate --offset 2 --output runs/offset2
dualview_isac evaluate --split transfer --output runs/transfer
```

Use each fitted control's own checkpoint at evaluation. `--label-ratio` reveals a seeded class-stratified fraction of the available training labels; unrevealed pairs still contribute self-supervision. The supervised controls use only the chosen classification head. Missing-view stress removes one randomly chosen view on the requested fraction of windows and uses the corresponding single-view prediction. Offset stress moves radar tokens with zero-filled exposed boundaries; one default token spans 160 ms. Keep offset and missingness conditions separate unless a combined stress is intended.

Training flags are `--epochs`, `--batch-size`, `--lr`, `--dimension`, `--temperature`, `--radius`, `--temporal-weight`, `--view-weight`, `--seed`, and `--device`. The default local radius is two tokens. Each view uses a convolutional front end and GRU, with no shared parameters. Temporal contrastive loss pools allowed offsets by differentiable log-sum-exp while treating other paired windows as negatives. ViewDrop moves single-view embeddings toward a detached fused target and supervises available single-view heads.

## Inference and outputs

`model.pt` stores weights and embedding dimension. The Python model's `predict(csi=...)` and `predict(radar=...)` paths execute only the available encoder and head; `predict(csi=..., radar=...)` uses fusion. They consume normalized feature tensors shaped `[batch,tokens,channels,32,32]`, with four CSI channels or two radar channels.

Evaluation writes `metrics.json`, `predictions.npz`, `confusion.csv` and `confusion.png`. Metrics include six-class macro-F1, balanced accuracy, group/room/device macro-F1, model parameters and measured milliseconds per window. Labels `-1` are excluded from recognition metrics. Single-view latency executes the selected path; missing-view batch stress times the full model plus path selection. Timing excludes prepared-data loading and synchronizes the GPU.

## Implementation choices

The manuscript does not specify exact RF FFT sizes, optimizer settings or temporal layers. This implementation uses each recorded chirp/frame's available FFT axes, fixed-size interpolated feature maps, a 128-dimensional representation, AdamW at 0.001 and 50 epochs. CSI phase calibration removes per-packet subcarrier slope and intercept before adjacent-antenna differencing. These choices define a runnable signal pipeline without assigning the manuscript's parameter or latency values to this network. No missing waveform is reconstructed. Test partitions do not contribute normalization or labels to training.
