"""Learnable time embedding from t-PatchGNN (ICML 2024).

Matches ``tPatchGNN.LearnableTE``: linear scale + ``sin(W t + b)`` periodic terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnableTimeEmbedding(nn.Module):
    def __init__(self, te_dim: int) -> None:
        super().__init__()
        if te_dim < 2:
            raise ValueError("te_dim must be >= 2 (1 linear + >=1 periodic)")
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, te_dim - 1)

    def forward(self, tt: torch.Tensor) -> torch.Tensor:
        """tt: (..., 1) normalized time in [0, 1]. Returns (..., te_dim)."""
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], dim=-1)
