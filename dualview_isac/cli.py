import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.nn import functional as F

from .method import DualViewISAC, contrastive, temporal_contrastive


def resize_map(values):
    tensor = torch.as_tensor(np.asarray(values, np.float32))[None, None]
    return F.interpolate(tensor, size=(32, 32), mode="bilinear", align_corners=False).numpy()[0, 0]


def csi_features(signal, tokens):
    result = []
    for chunk in np.array_split(signal, tokens):
        if len(chunk) < 2:
            raise ValueError("Each CSI token requires at least two samples")
        amplitude = np.abs(chunk).mean(0)
        phase = np.unwrap(np.angle(chunk), axis=-1)
        axis = np.arange(phase.shape[-1])
        centered = axis - axis.mean()
        slope = ((phase - phase.mean(-1, keepdims=True)) * centered).sum(-1, keepdims=True) / max(np.sum(centered ** 2), 1)
        calibrated = phase - phase.mean(-1, keepdims=True) - slope * centered
        difference = np.diff(calibrated, axis=1, prepend=calibrated[:, :1]).mean(0)
        antennas = chunk.transpose(1, 0, 2).reshape(chunk.shape[1], -1)
        correlation = np.abs(antennas @ antennas.conj().T) / np.maximum(np.linalg.norm(antennas, axis=1)[:, None] * np.linalg.norm(antennas, axis=1)[None], 1e-8)
        doppler = np.log1p(np.abs(np.fft.fftshift(np.fft.fft(chunk - chunk.mean(0), axis=0), axes=0)).mean(1))
        result.append(np.stack([resize_map(amplitude), resize_map(difference), resize_map(correlation), resize_map(doppler)]))
    return np.array(result, np.float32)


def radar_features(echo, tokens):
    range_doppler = np.fft.fftshift(np.fft.fft2(echo, axes=(-2, -1)), axes=-2)
    maps = np.log1p(np.abs(range_doppler))
    result = []
    for chunk in np.array_split(maps, tokens):
        if not len(chunk):
            raise ValueError("Radar record has fewer frames than tokens")
        result.append(np.stack([resize_map(chunk.mean(0)), resize_map(chunk.sum(-1).T)]))
    return np.array(result, np.float32)


def prepare(args):
    manifest = Path(args.manifest)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    seen, packed = {}, {}
    for row in rows:
        group, split = str(row["group"]), row["split"]
        if group in seen and seen[group] != split:
            raise ValueError("Subject/session group crosses partitions")
        seen[group] = split
        raw = np.load(manifest.resolve().parent / row["file"], allow_pickle=False)
        csi, radar = raw["csi"], raw["radar"]
        if csi.ndim != 3 or radar.ndim != 3 or not np.iscomplexobj(csi) or not np.iscomplexobj(radar):
            raise ValueError("Expected complex CSI[T,antenna,subcarrier] and radar[frame,chirp,fast_sample]")
        duration = min(len(csi) / float(raw["csi_fs"]), len(radar) / float(raw["radar_fs"]))
        stride = args.seconds / 2 if split == "train" else args.seconds
        for start in np.arange(0, duration - args.seconds + 1e-7, stride):
            c = csi[round(start * float(raw["csi_fs"])):round((start + args.seconds) * float(raw["csi_fs"]))]
            r = radar[round(start * float(raw["radar_fs"])):round((start + args.seconds) * float(raw["radar_fs"]))]
            packed.setdefault(split, []).append((csi_features(c, args.tokens), radar_features(r, args.tokens), int(row.get("label", -1)), group, str(row.get("room", "unknown")), str(row.get("device", "unknown"))))
    training = packed["train"]
    stats = []
    for view in range(2):
        values = np.stack([s[view] for s in training])
        stats.append((values.mean((0, 1, 3, 4)), np.maximum(values.std((0, 1, 3, 4)), 1e-6)))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for split, samples in packed.items():
        c, r, y, group, room, device = zip(*samples)
        c = (np.array(c) - stats[0][0][None, None, :, None, None]) / stats[0][1][None, None, :, None, None]
        r = (np.array(r) - stats[1][0][None, None, :, None, None]) / stats[1][1][None, None, :, None, None]
        np.savez_compressed(out / f"{split}.npz", csi=c.astype(np.float32), radar=r.astype(np.float32), y=y, group=group, room=room, device=device)
    np.savez_compressed(out / "normalization.npz", csi_mean=stats[0][0], csi_std=stats[0][1], radar_mean=stats[1][0], radar_std=stats[1][1])


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = np.load(Path(args.data) / "train.npz", allow_pickle=False)
    labels = data["y"].copy()
    for c in np.unique(labels[labels >= 0]):
        index = rng.permutation(np.flatnonzero(labels == c))
        labels[index[max(1, round(len(index) * args.label_ratio)):]] = -1
    net = DualViewISAC(args.dimension).to(args.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    for _ in range(args.epochs):
        net.train()
        order = rng.permutation(len(labels))
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            if len(index) < 2 and not args.supervised_only:
                continue
            c, r = [torch.as_tensor(data[k][index], device=args.device) for k in ["csi", "radar"]]
            out = net(c, r)
            loss = out["full"].sum() * 0
            if not args.supervised_only:
                loss = loss + contrastive(out["zc"], out["zr"], args.temperature)
                if not args.no_alignment:
                    loss = loss + args.temporal_weight * temporal_contrastive(out["hc"], out["hr"], args.radius, args.temperature)
                if not args.no_viewdrop:
                    loss = loss + args.view_weight * ((out["zc"] - out["zf"].detach()).square().mean() + (out["zr"] - out["zf"].detach()).square().mean())
            y = torch.as_tensor(labels[index], device=args.device)
            known = y >= 0
            if args.supervised_only and not known.any():
                continue
            if known.any():
                views = [args.view] if args.supervised_only else (["full"] if args.no_viewdrop else ["full", "csi", "radar"])
                loss = loss + sum(F.cross_entropy(out[view][known], y[known]) for view in views) / len(views)
            if not loss.requires_grad:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    outpath = Path(args.output)
    outpath.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "dimension": args.dimension}, outpath / "model.pt")


def shifted(x, offset):
    if offset == 0:
        return x
    out = torch.zeros_like(x)
    if offset > 0:
        out[:, offset:] = x[:, :-offset]
    else:
        out[:, :offset] = x[:, -offset:]
    return out


def evaluate(args):
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    net = DualViewISAC(saved["dimension"]).to(args.device)
    net.load_state_dict(saved["model"])
    net.eval()
    data = np.load(Path(args.data) / f"{args.split}.npz", allow_pickle=False)
    probability, labels, elapsed = [], [], []
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        for start in range(0, len(data["y"]), args.batch_size):
            index = np.arange(start, min(start + args.batch_size, len(data["y"])))
            c, r = [torch.as_tensor(data[k][index], device=args.device) for k in ["csi", "radar"]]
            r = shifted(r, args.offset)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            begin = perf_counter()
            if args.view == "full":
                output = net(c, r)
                p = output["full"].softmax(-1)
                missing = rng.random(len(index)) < args.missing_ratio
                choose_csi = rng.random(len(index)) < 0.5
                for i in range(len(index)):
                    if missing[i]:
                        p[i] = output["csi" if choose_csi[i] else "radar"][i].softmax(-1)
            else:
                p = net.predict(csi=c if args.view == "csi" else None, radar=r if args.view == "radar" else None).softmax(-1)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed.append(perf_counter() - begin)
            probability.append(p.cpu().numpy())
            labels.append(data["y"][index])
    probability, labels = np.concatenate(probability), np.concatenate(labels)
    known = labels >= 0
    pred = probability.argmax(1)
    metrics = {"macro_f1": float(f1_score(labels[known], pred[known], labels=range(6), average="macro", zero_division=0)),
               "balanced_accuracy": float(balanced_accuracy_score(labels[known], pred[known])),
               "model_parameters": sum(p.numel() for p in net.parameters()), "ms_per_window": 1000 * sum(elapsed) / len(labels)}
    for field in ["room", "device", "group"]:
        metrics[field + "_macro_f1"] = {str(g): float(f1_score(labels[(data[field] == g) & known], pred[(data[field] == g) & known], labels=range(6), average="macro", zero_division=0)) for g in np.unique(data[field][known])}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    np.savez_compressed(out / "predictions.npz", probability=probability, y=labels, group=data["group"])
    cm = confusion_matrix(labels[known], pred[known], labels=range(6))
    np.savetxt(out / "confusion.csv", cm, delimiter=",", fmt="%d")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set(xlabel="Prediction", ylabel="Reference")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="data/prepared")
    p.add_argument("--seconds", type=float, default=2.56)
    p.add_argument("--tokens", type=int, default=16)
    p.set_defaults(function=prepare)
    for command in ["train", "evaluate"]:
        p = sub.add_parser(command)
        p.add_argument("--data", default="data/prepared")
        p.add_argument("--output", default=f"runs/{command}")
        p.add_argument("--device", default="cpu")
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--view", choices=["full", "csi", "radar"], default="full")
        if command == "train":
            p.add_argument("--epochs", type=int, default=50)
            p.add_argument("--dimension", type=int, default=128)
            p.add_argument("--radius", type=int, default=2)
            for key, value in [("lr", 1e-3), ("label-ratio", 1), ("temperature", 0.1), ("temporal-weight", 0.5), ("view-weight", 1)]:
                p.add_argument("--" + key, type=float, default=value)
            for flag in ["no-alignment", "no-viewdrop", "supervised-only"]:
                p.add_argument("--" + flag, action="store_true")
            p.set_defaults(function=train)
        else:
            p.add_argument("--checkpoint", default="runs/train/model.pt")
            p.add_argument("--split", default="test")
            p.add_argument("--offset", type=int, default=0)
            p.add_argument("--missing-ratio", type=float, default=0)
            p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)
