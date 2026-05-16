from __future__ import annotations

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t.reshape(1, 1)).reshape(1, 1, -1)


class VectorField(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, condition_step: bool = False) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.condition_step = condition_step
        self.time = TimeEmbedding(dim)
        self.step = TimeEmbedding(dim) if condition_step else None
        self.net = nn.Sequential(
            nn.Linear(dim * (3 if condition_step else 2), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, step_size: torch.Tensor | None = None) -> torch.Tensor:
        time = self.time(t).expand_as(x)
        parts = [x, time]
        if self.condition_step:
            if step_size is None:
                step_size = x.new_tensor(1.0)
            parts.append(self.step(step_size).expand_as(x))
        return self.net(torch.cat(parts, dim=-1))


class EulerFlow(nn.Module):
    def __init__(
        self, dim: int, steps: int = 3, hidden_dim: int | None = None, mode: str = "euler", step_size: float = 1.0
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if mode not in {"euler", "direct", "shortcut"}:
            raise ValueError("mode must be euler, direct, or shortcut")
        self.steps = steps
        self.mode = mode
        self.step_size = step_size
        self.field = VectorField(dim, hidden_dim, condition_step=(mode == "shortcut"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in {"direct", "shortcut"}:
            d = x.new_tensor(self.step_size)
            return x + d * self.field(x, x.new_tensor(0.0), d)
        dt = 1.0 / self.steps
        out = x
        for i in range(self.steps):
            t = out.new_tensor(i / self.steps)
            out = out + dt * self.field(out, t)
        return out
