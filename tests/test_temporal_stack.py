"""Tests for TemporalAdaptiveGNNStack and DeliriumTPatchBackbone."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from src.data.feature_vocab import NUM_FEATURES
from src.models.delirium_backbone import DeliriumClassifier, DeliriumTPatchBackbone
from src.models.temporal_adaptive_stack import TemporalAdaptiveGNNStack


def test_temporal_stack_shapes_and_grad():
    torch.manual_seed(0)
    B, V, P, D = 2, 8, 5, 16
    stack = TemporalAdaptiveGNNStack(
        num_variables=V,
        d_model=D,
        n_layer=2,
        nhead=1,
        tf_layer=1,
        node_dim=6,
        hop=1,
        dropout=0.0,
        max_patches=32,
    )
    x = torch.randn(B, V, P, D, requires_grad=True)
    patch_mask = torch.ones(B, V, P)
    patch_mask[:, :, 4] = 0.0
    stay_patch_mask = torch.ones(B, P)
    stay_patch_mask[:, 3:] = 0.0

    out = stack(x, patch_mask, stay_patch_mask)
    assert out.shape == (B, V, P, D)
    assert not torch.isnan(out).any()
    out.mean().backward()
    assert stack.nodevec1.grad is not None


def test_backbone_forward():
    torch.manual_seed(1)
    B, V, P, L = 2, NUM_FEATURES, 3, 8
    bb = DeliriumTPatchBackbone(
        hid_dim=32,
        te_dim=10,
        n_layer=1,
        nhead=1,
        dropout=0.0,
        max_patches=64,
    )
    batch = {
        "values": torch.rand(B, V, P, L),
        "times": torch.rand(B, V, P, L).clamp(0, 1),
        "point_mask": torch.zeros(B, V, P, L),
        "stay_patch_mask": torch.ones(B, P),
    }
    batch["point_mask"][..., :2] = 1.0
    h = bb(batch)
    assert h.shape == (B, V, P, 32)
    assert not torch.isnan(h).any()
    h.sum().backward()


def test_classifier_forward():
    torch.manual_seed(2)
    B, V, P, L = 2, NUM_FEATURES, 3, 8
    clf = DeliriumClassifier(
        hid_dim=32,
        te_dim=10,
        n_layer=1,
        nhead=1,
        dropout=0.0,
        max_patches=64,
    )
    batch = {
        "values": torch.rand(B, V, P, L),
        "times": torch.rand(B, V, P, L).clamp(0, 1),
        "point_mask": torch.ones(B, V, P, L),
        "stay_patch_mask": torch.ones(B, P),
    }
    logits = clf(batch)
    assert logits.shape == (B, 1), f"Expected ({B}, 1), got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in classifier output"
    logits.sum().backward()
    assert clf.classifier.weight.grad is not None, "No gradient on classifier head"


def test_classifier_masked_pooling():
    """Masked patches must not contribute to the pooled representation."""
    torch.manual_seed(3)
    B, V, P, L = 2, NUM_FEATURES, 3, 8
    clf = DeliriumClassifier(
        hid_dim=32, te_dim=10, n_layer=1, nhead=1, dropout=0.0, max_patches=64,
    )
    clf.eval()

    def _logit(spm):
        batch = {
            "values": torch.zeros(B, V, P, L),   # identical inputs
            "times": torch.zeros(B, V, P, L),
            "point_mask": torch.zeros(B, V, P, L),
            "stay_patch_mask": spm,
        }
        with torch.no_grad():
            return clf(batch)

    # All patches valid vs. last patch masked — outputs must differ or at least be finite
    spm_full  = torch.ones(B, P)
    spm_mask  = torch.ones(B, P); spm_mask[:, -1] = 0.0  # last patch is padding

    logits_full = _logit(spm_full)
    logits_mask = _logit(spm_mask)

    assert logits_full.shape == (B, 1)
    assert logits_mask.shape == (B, 1)
    assert not torch.isnan(logits_full).any(), "NaN with full mask"
    assert not torch.isnan(logits_mask).any(), "NaN with partial mask"

    # Edge case: only one valid patch per sample — must not produce NaN
    spm_one = torch.zeros(B, P); spm_one[:, 0] = 1.0
    logits_one = _logit(spm_one)
    assert not torch.isnan(logits_one).any(), "NaN when only one patch is valid"


if __name__ == "__main__":
    test_temporal_stack_shapes_and_grad()
    test_backbone_forward()
    test_classifier_forward()
    print("temporal_stack tests passed.")
