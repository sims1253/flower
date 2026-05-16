from __future__ import annotations

import torch
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward
from flower.models.memory import MemoryRead


class SummaryMemoryBlock(nn.Module):
    def __init__(self, config: ModelConfig, memory_read: MemoryRead | None = None) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = memory_read or MemoryRead(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout)
        self.token_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model)
        )
        self.mem_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model)
        )
        self.update = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model)
        )
        self.agg_query = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.perceiver_latents = nn.Parameter(torch.randn(1, config.memory_slots, config.d_model) * 0.02)
        self.perceiver = nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)
        self.short_project = nn.Linear(config.d_model, config.d_model)
        if config.memory_aggregation not in {"sum", "mean", "max", "attention", "orthogonal"}:
            raise ValueError("memory_aggregation must be sum, mean, max, attention, or orthogonal")
        if config.summary_style not in {"deepsets", "perceiver"}:
            raise ValueError("summary_style must be deepsets or perceiver")

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        slots = self.config.memory_slots + (self.config.short_memory_slots if self.config.hierarchical_memory else 0)
        return x.new_zeros(x.shape[0], slots, self.config.d_model)

    def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.config.memory_aggregation
        if mode == "sum":
            return x.sum(dim=1, keepdim=True)
        # `orthogonal` shares the max-pool aggregation with `max`; the orthogonality
        # constraint is applied later in `_update_memory` as a projection on the
        # candidate update vector, not on the aggregation step.
        if mode == "max" or mode == "orthogonal":
            return x.max(dim=1, keepdim=True).values
        if mode == "attention":
            q = self.agg_query.expand(x.shape[0], -1, -1)
            weights = torch.softmax(q @ x.transpose(1, 2) / (x.shape[-1] ** 0.5), dim=-1)
            return weights @ x
        return x.mean(dim=1, keepdim=True)

    @staticmethod
    def _orthogonal_residual(update: torch.Tensor, memory: torch.Tensor, eps: float) -> torch.Tensor:
        """Project `update` onto the orthogonal complement of `memory` (per batch).

        update: (B, S, D)  candidate write to each memory slot
        memory: (B, S, D)  current contents of those slots
        Returns the component of `update` that's orthogonal to the row span of `memory`,
        so the additive write minimally overlaps with what's already stored (LATTICE A1).
        """
        # Normalise rows of memory to get an approximate orthonormal basis. For S<=D this
        # works as a cheap stand-in for full Gram-Schmidt; we project each update row
        # against each normalised memory row independently.
        mem_norm = memory / (memory.norm(dim=-1, keepdim=True) + eps)  # (B, S, D)
        # Coefficients of update along each memory row: (B, S, 1) via batched dot product.
        coeff = (update * mem_norm).sum(dim=-1, keepdim=True)
        return update - coeff * mem_norm

    def _update_memory(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.config.summary_style == "perceiver":
            latents = self.perceiver_latents.expand(x.shape[0], -1, -1)
            long_update, _ = self.perceiver(latents, x, x, need_weights=False)
            token_summary = long_update
        else:
            token_summary = self._aggregate(x).expand(x.shape[0], self.config.memory_slots, x.shape[-1])
        long_mem = memory[:, : self.config.memory_slots]
        combined = self.token_mlp(token_summary) + self.mem_mlp(long_mem)
        candidate_update = self.update(combined) / max(1, self.config.num_layers)
        if self.config.memory_aggregation == "orthogonal":
            candidate_update = self._orthogonal_residual(candidate_update, long_mem, self.config.orthogonal_eps)
        long_mem = long_mem + candidate_update
        if not self.config.hierarchical_memory:
            return long_mem
        short = self.short_project(x[:, -self.config.short_memory_slots :])
        if short.shape[1] < self.config.short_memory_slots:
            short = torch.nn.functional.pad(short, (0, 0, self.config.short_memory_slots - short.shape[1], 0))
        return torch.cat([long_mem, short], dim=1)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory is None:
            memory = self._initial_memory(x)
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))
        if self.config.memory_update_frequency <= 1 or x.shape[1] % self.config.memory_update_frequency == 0:
            memory = self._update_memory(memory, x)
        return x, memory


def build_summary_memory_model(config: ModelConfig) -> CausalLM:
    blocks = [SummaryMemoryBlock(config) for _ in range(config.num_layers)]
    return CausalLM(config, blocks)
