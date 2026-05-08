from __future__ import annotations

import torch
from torch import nn


class CouplingNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AffineCouplingLayer(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Coupling dimension must be even")
        half = dim // 2
        hidden_dim = hidden_dim or dim * 2
        self.first = CouplingNet(half + cond_dim, half * 2, hidden_dim)
        self.second = CouplingNet(half + cond_dim, half * 2, hidden_dim)

    @staticmethod
    def _scale_shift(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = raw.chunk(2, dim=-1)
        return torch.tanh(scale) * 0.5, shift

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z1, z2 = z.chunk(2, dim=-1)
        s1, t1 = self._scale_shift(self.first(torch.cat([z2, cond], dim=-1)))
        z1p = z1 * torch.exp(s1) + t1
        s2, t2 = self._scale_shift(self.second(torch.cat([z1p, cond], dim=-1)))
        z2p = z2 * torch.exp(s2) + t2
        return torch.cat([z1p, z2p], dim=-1)

    def inverse(self, zp: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z1p, z2p = zp.chunk(2, dim=-1)
        s2, t2 = self._scale_shift(self.second(torch.cat([z1p, cond], dim=-1)))
        z2 = (z2p - t2) * torch.exp(-s2)
        s1, t1 = self._scale_shift(self.first(torch.cat([z2, cond], dim=-1)))
        z1 = (z1p - t1) * torch.exp(-s1)
        return torch.cat([z1, z2], dim=-1)


class ConditionalCouplingFlow(nn.Module):
    def __init__(self, dim: int, cond_dim: int, layers: int = 2, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [AffineCouplingLayer(dim, cond_dim, hidden_dim) for _ in range(layers)]
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = z
        for layer in self.layers:
            out = layer(out.flip(-1), cond).flip(-1)
        return out

    def inverse(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = z
        for layer in reversed(self.layers):
            out = layer.inverse(out.flip(-1), cond).flip(-1)
        return out
