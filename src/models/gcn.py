"""Graph convolution blocks from t-PatchGNN (device-agnostic)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NConv(nn.Module):
    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """x (B, F, N, M), A (B, M, N, N) -> (B, F, N, M)."""
        return torch.einsum("bfnm,bmnv->bfvm", x, a).contiguous()


class Conv1x1(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        dropout: float,
        support_len: int = 1,
        order: int = 2,
    ) -> None:
        super().__init__()
        self.nconv = NConv()
        c_cat = (order * support_len + 1) * c_in
        self.mlp = Conv1x1(c_cat, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x: torch.Tensor, support: list[torch.Tensor]) -> torch.Tensor:
        """x (B, F, N, M); each support[i] (B, M, N, N)."""
        out: list[torch.Tensor] = [x]
        for adj in support:
            x1 = self.nconv(x, adj)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, adj)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.relu(h)
