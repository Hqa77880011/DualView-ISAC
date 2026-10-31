"""Finite learning, checkpoint and metric checks; no recognition claim."""

import io
import math
import unittest

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.nn import functional as F

from dualview_isac.cli import shifted
from dualview_isac.method import DualViewISAC, contrastive, temporal_contrastive


class MethodTests(unittest.TestCase):
    def test_learning_and_checkpoint(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        csi, radar = torch.randn(2, 4, 4, 16, 16), torch.randn(2, 4, 2, 16, 16)
        model = DualViewISAC(8)
        output = model(csi, radar)
        loss = contrastive(output["zc"], output["zr"], 0.1)
        loss += 0.5 * temporal_contrastive(output["hc"], output["hr"], 1, 0.1)
        loss += sum((output[v] - output["zf"].detach()).square().mean() for v in ["zc", "zr"])
        loss += sum(F.cross_entropy(output[v], torch.tensor([0, 1])) for v in ["csi", "radar", "full"])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for prefix in ["csi.front", "csi.temporal", "radar.front", "radar.temporal",
                       "fusion", "csi_head", "radar_head", "fused_head"]:
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                                for name, p in model.named_parameters() if name.startswith(prefix)), prefix)
        before = model.csi.front[0].weight.detach().clone()
        torch.optim.Adam(model.parameters(), lr=0.01).step()
        self.assertFalse(torch.equal(before, model.csi.front[0].weight))
        model.eval()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = DualViewISAC(8).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        for name, value in model(csi, radar).items():
            torch.testing.assert_close(value, restored(csi, radar)[name], rtol=1e-6, atol=1e-7)

    def test_offsets_and_hand_computed_metrics(self):
        values = torch.tensor([0., 2., 4.]).reshape(1, 3, 1, 1, 1)
        torch.testing.assert_close(shifted(values, 0.5).flatten(), torch.tensor([0., 1., 3.]))
        torch.testing.assert_close(shifted(values, -1).flatten(), torch.tensor([2., 4., 0.]))
        torch.testing.assert_close(shifted(values, 5), torch.zeros_like(values))
        self.assertAlmostEqual(float(contrastive(torch.eye(2), torch.eye(2), 1)), math.log1p(math.exp(-1)), places=6)
        self.assertAlmostEqual(f1_score([0, 1], [0, 0], labels=np.arange(6), average="macro", zero_division=0), 1 / 9)
        self.assertEqual(balanced_accuracy_score([0, 1], [0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
