"""Smoke tests for patch dataset collation and PatchTTCNEncoder."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow `python tests/test_patch_encoder.py` from project root (adds repo root to path).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data.feature_vocab import FEATURE_NAMES, NUM_FEATURES
from src.data.patch_dataset import ICUPatchDataset, collate_patches
from src.models.patch_encoder import PatchTTCNEncoder


def test_patch_encoder_shapes_and_gradient():
    torch.manual_seed(0)
    B, V, P, L = 2, NUM_FEATURES, 4, 8
    hid, te_d = 32, 10
    enc = PatchTTCNEncoder(hid_dim=hid, te_dim=te_d)
    values = torch.rand(B, V, P, L)
    times = torch.rand(B, V, P, L).clamp(0, 1)
    point_mask = torch.zeros(B, V, P, L)
    point_mask[..., :3] = 1.0

    out, pm = enc(values, times, point_mask)
    assert out.shape == (B, V, P, hid)
    assert pm.shape == (B, V, P)

    loss = out.mean()
    loss.backward()
    assert enc.ttcn.filter_generators[0].weight.grad is not None


def test_dataset_and_collate_roundtrip():
    d = tempfile.mkdtemp()
    root = Path(d)
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "los_hours": [20.0, 10.0],
            "label": [0, 1],
        }
    )
    rows = []
    for sid, nh in [(1, 20), (2, 10)]:
        for h in range(nh):
            for fi, fn in enumerate(FEATURE_NAMES[:3]):
                if h % (fi + 2) == 0:
                    rows.append(
                        {
                            "stay_id": sid,
                            "hour_offset": h,
                            "feature_name": fn,
                            "value": float(h + fi * 0.1),
                        }
                    )
    feats = pd.DataFrame(rows)
    cohort.to_csv(root / "cohort.csv", index=False)
    feats.to_csv(root / "features_hourly.csv", index=False)

    ds = ICUPatchDataset(root / "cohort.csv", root / "features_hourly.csv")
    assert len(ds) == 2
    b = collate_patches([ds[0], ds[1]])
    assert b["values"].shape[0] == 2
    assert b["values"].shape[1] == NUM_FEATURES

    enc = PatchTTCNEncoder()
    out, _ = enc(b["values"], b["times"], b["point_mask"])
    assert out.shape == (2, NUM_FEATURES, b["values"].shape[2], 32)
    assert not torch.isnan(out).any()


def test_prelocf_changes_mask():
    d = tempfile.mkdtemp()
    root = Path(d)
    cohort = pd.DataFrame({"stay_id": [1], "los_hours": [8.0], "label": [0]})
    cohort.to_csv(root / "cohort.csv", index=False)
    # LOCF-style: hour 0 and 7 only for heart_rate in prelocf
    pre = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "hour_offset": [0, 7],
            "feature_name": ["heart_rate", "heart_rate"],
            "value": [70.0, 72.0],
        }
    )
    pre.to_csv(root / "features_hourly_prelocf.csv", index=False)
    # Post-LOCF has all hours filled
    locf_rows = [{"stay_id": 1, "hour_offset": h, "feature_name": "heart_rate", "value": 70.0} for h in range(8)]
    pd.DataFrame(locf_rows).to_csv(root / "features_hourly.csv", index=False)

    ds = ICUPatchDataset(
        root / "cohort.csv",
        root / "features_hourly.csv",
        prelocf_features_path=root / "features_hourly_prelocf.csv",
    )
    s = ds[0]
    hr_i = FEATURE_NAMES.index("heart_rate")
    # Only two timesteps observed in patch 0
    assert float(s["point_mask"][hr_i, 0].sum()) == 2.0


if __name__ == "__main__":
    test_patch_encoder_shapes_and_gradient()
    test_dataset_and_collate_roundtrip()
    test_prelocf_changes_mask()
    print("All patch_encoder smoke tests passed.")
