import torch
from torch import nn
from torch.nn import functional as F


class ViewEncoder(nn.Module):
    def __init__(self, channels, dimension):
        super().__init__()
        self.front = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(), nn.MaxPool2d(2),
                                   nn.Conv2d(32, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dimension))
        self.temporal = nn.GRU(dimension, dimension, batch_first=True)

    def forward(self, x):
        batch, time, channels, height, width = x.shape
        tokens = self.front(x.reshape(batch * time, channels, height, width)).reshape(batch, time, -1)
        tokens, _ = self.temporal(tokens)
        return tokens, F.normalize(tokens.mean(1), dim=-1)


class DualViewISAC(nn.Module):
    def __init__(self, dimension=128, classes=6):
        super().__init__()
        self.csi = ViewEncoder(4, dimension)
        self.radar = ViewEncoder(2, dimension)
        self.fusion = nn.Sequential(nn.Linear(2 * dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.csi_head, self.radar_head, self.fused_head = [nn.Linear(dimension, classes) for _ in range(3)]

    def forward(self, csi, radar):
        hc, zc = self.csi(csi)
        hr, zr = self.radar(radar)
        zf = F.normalize(self.fusion(torch.cat([zc, zr], -1)), dim=-1)
        return {"hc": hc, "hr": hr, "zc": zc, "zr": zr, "zf": zf,
                "csi": self.csi_head(zc), "radar": self.radar_head(zr), "full": self.fused_head(zf)}

    def predict(self, csi=None, radar=None):
        if radar is None and csi is not None:
            return self.csi_head(self.csi(csi)[1])
        if csi is None and radar is not None:
            return self.radar_head(self.radar(radar)[1])
        if csi is None or radar is None:
            raise ValueError("At least one RF view is required")
        return self(csi, radar)["full"]


class EarlyFusionISAC(nn.Module):
    """Supervised early fusion of aligned CSI/radar feature maps before one CNN/GRU."""
    def __init__(self, dimension=128, classes=6):
        super().__init__()
        self.encoder = ViewEncoder(6, dimension)
        self.head = nn.Linear(dimension, classes)

    def forward(self, csi, radar):
        return {"full": self.head(self.encoder(torch.cat([csi, radar], dim=2))[1])}

    def predict(self, csi=None, radar=None):
        if csi is None and radar is None:
            raise ValueError("At least one RF view is required")
        if csi is None:
            csi = radar.new_zeros((*radar.shape[:2], 4, *radar.shape[3:]))
        if radar is None:
            radar = csi.new_zeros((*csi.shape[:2], 2, *csi.shape[3:]))
        return self(csi, radar)["full"]


def matched_early_dimension(reference_dimension, classes=6):
    """Choose the nearest early-fusion width to the complete DualView parameter budget."""
    target = sum(p.numel() for p in DualViewISAC(reference_dimension, classes).parameters())
    # ViewEncoder(6,d) + classifier: 6*d**2 + (71+classes)*d + 20320+classes.
    linear, constant = 71 + classes, 20320 + classes
    root = (-linear + (linear**2 + 24 * (target - constant))**0.5) / 12
    choices = [max(1, int(root)), max(1, int(root) + 1)]
    return min(choices, key=lambda d: abs(6*d*d + linear*d + constant - target))


def contrastive(zc, zr, temperature):
    similarity = F.normalize(zc, dim=-1) @ F.normalize(zr, dim=-1).T / temperature
    labels = torch.arange(len(zc), device=zc.device)
    return (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels)) / 2


def temporal_contrastive(hc, hr, radius, temperature):
    hc, hr = F.normalize(hc, dim=-1), F.normalize(hr, dim=-1)
    batch, time, _ = hc.shape
    similarity = torch.einsum("itd,jsd->itjs", hc, hr) / temperature
    index = torch.arange(time, device=hc.device)
    local = (index[:, None] - index[None]).abs() <= radius
    similarity = similarity.masked_fill(~local[None, :, None, :], -1e9)
    diagonal = similarity[torch.arange(batch, device=hc.device), :, torch.arange(batch, device=hc.device), :]
    forward = similarity.logsumexp((2, 3)) - diagonal.logsumexp(-1)
    reverse = similarity.logsumexp((0, 1)) - diagonal.logsumexp(1)
    return (forward.mean() + reverse.mean()) / 2
