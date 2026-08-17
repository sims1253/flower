from __future__ import annotations

import torch
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward
from flower.models.memory import (
    MemoryRead,
    SDPCrossAttention,
    causal_last_tokens,
    causal_prefix_attention,
    causal_running_mean,
)


class SummaryMemoryBlock(nn.Module):
    def __init__(self, config: ModelConfig, memory_read: MemoryRead | None = None) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = memory_read or MemoryRead(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout, config=config)
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
        # SDP cross-attention (compile-clean) replaces nn.MultiheadAttention: MHA
        # graph-breaks under torch.compile and OOMs at long context. See
        # SDPCrossAttention docstring / NEXT_IDEAS.md section 4. Same params, same
        # cross-attention math (Q=latents, K=V=window, no causal mask).
        self.perceiver = SDPCrossAttention(config)
        self.short_project = nn.Linear(config.d_model, config.d_model)
        if config.memory_aggregation not in {"sum", "mean", "max", "attention", "orthogonal"}:
            raise ValueError("memory_aggregation must be sum, mean, max, attention, or orthogonal")
        if config.summary_style not in {"deepsets", "perceiver"}:
            raise ValueError("summary_style must be deepsets or perceiver")

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        slots = self.config.memory_slots + (self.config.short_memory_slots if self.config.hierarchical_memory else 0)
        return x.new_zeros(x.shape[0], slots, self.config.d_model)

    def _initial_memory_causal(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, S_total, D) per-position memory state for causal_memory=True."""
        slots = self.config.memory_slots + (self.config.short_memory_slots if self.config.hierarchical_memory else 0)
        return x.new_zeros(x.shape[0], x.shape[1], slots, self.config.d_model)

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

    def _aggregate_causal(self, x: torch.Tensor) -> torch.Tensor:
        """Per-position prefix aggregate — causal analogue of ``_aggregate``.

        Returns (B, T, D): out[:, t] aggregates x[:, :t+1] only, so the write
        at t is a function of tokens <= t. `orthogonal` shares the max-pool
        aggregation with `max`, as in ``_aggregate``.
        """
        mode = self.config.memory_aggregation
        if mode == "sum":
            return torch.cumsum(x, dim=1)
        if mode == "max" or mode == "orthogonal":
            return torch.cummax(x, dim=1).values
        if mode == "attention":
            # Same q @ x^T / sqrt(D) scoring as the legacy attention
            # aggregation; causal_prefix_attention restricts the softmax to
            # tokens <= t (the learned global query is the single "latent").
            q = self.agg_query.expand(x.shape[0], -1, -1)  # (B, 1, D)
            scores = q @ x.transpose(1, 2) / (x.shape[-1] ** 0.5)  # (B, 1, T)
            scores = scores.unsqueeze(1)  # (B, H=1, P=1, T)
            out = causal_prefix_attention(scores, x.unsqueeze(1))  # (B, 1, T, 1, D)
            return out.squeeze(1).squeeze(2)  # (B, T, D)
        return causal_running_mean(x)

    @staticmethod
    def _orthogonal_residual(update: torch.Tensor, memory: torch.Tensor, eps: float) -> torch.Tensor:
        """Per-row Gram projection: orthogonalise each update row against its
        OWN memory row only.

        update: (B, S, D)  candidate write to each memory slot
        memory: (B, S, D)  current contents of those slots

        What is actually computed (kept as-is; changing it would need a flag):
        `coeff` is (B, S, 1) — one coefficient per row — so update row s is
        projected against the eps-regularised unit-normalised memory row s
        (the slot that row writes into) and nothing else. The result is
        orthogonal to that same-slot memory row, NOT to the row span of the
        whole memory: rows s' != s are never seen by row s, so the residual
        can still overlap other slots' contents. A true projection onto the
        orthogonal complement of the memory's row span would require the
        (B, S, S) contraction coeff = update @ mem_norm^T (a Gram matrix over
        all pairs of memory rows) followed by a solved subtraction across all
        rows.

        Works unchanged on the causal (B, T, S, D) layout: the projection is
        pointwise over the last dim.
        """
        # Normalise rows of memory to get an approximate per-row basis. For
        # each slot s we project update row s against its own normalised
        # memory row only (see docstring — this is NOT a full row-span /
        # Gram-Schmidt projection over all S rows).
        mem_norm = memory / (memory.norm(dim=-1, keepdim=True) + eps)  # (B, S, D)
        # Coefficient of each update row along its OWN memory row: (B, S, 1)
        # via a per-row dot product.
        coeff = (update * mem_norm).sum(dim=-1, keepdim=True)
        return update - coeff * mem_norm

    def _update_memory(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.config.summary_style == "perceiver":
            latents = self.perceiver_latents.expand(x.shape[0], -1, -1)
            long_update = self.perceiver(latents, x)
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

    def _update_memory_causal(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """causal_memory=True write: memory is a per-position state (B, T, S_total, D).

        Identical recurrence to ``_update_memory`` (same MLPs, same scaling,
        no new parameters) but every quantity is per-position: the token
        summary at t aggregates tokens <= t, and the slot state read/written
        at t is memory[:, t]. The per-layer update is pointwise in t, so
        causality of the threaded memory follows by induction over layers.
        """
        if self.config.summary_style == "perceiver":
            latents = self.perceiver_latents.expand(x.shape[0], -1, -1)
            token_summary = self.perceiver.causal_forward(latents, x)  # (B, T, S, D)
        else:
            token_summary = self._aggregate_causal(x).unsqueeze(2)  # (B, T, 1, D)
            token_summary = token_summary.expand(-1, -1, self.config.memory_slots, -1)
        long_mem = memory[:, :, : self.config.memory_slots]
        combined = self.token_mlp(token_summary) + self.mem_mlp(long_mem)
        candidate_update = self.update(combined) / max(1, self.config.num_layers)
        if self.config.memory_aggregation == "orthogonal":
            # _orthogonal_residual is pointwise over the last dim — works
            # unchanged on the extra position axis.
            candidate_update = self._orthogonal_residual(candidate_update, long_mem, self.config.orthogonal_eps)
        long_mem = long_mem + candidate_update
        if not self.config.hierarchical_memory:
            return long_mem
        short = self.short_project(causal_last_tokens(x, self.config.short_memory_slots))  # (B, T, n, D)
        return torch.cat([long_mem, short], dim=2)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        causal = self.config.causal_memory
        if memory is None:
            memory = self._initial_memory_causal(x) if causal else self._initial_memory(x)
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)  # MemoryRead dispatches on memory.dim()
        x = x + self.ff(self.ln2(x))
        if self.config.memory_update_frequency <= 1 or x.shape[1] % self.config.memory_update_frequency == 0:
            memory = self._update_memory_causal(memory, x) if causal else self._update_memory(memory, x)
        return x, memory


def build_summary_memory_model(config: ModelConfig) -> CausalLM:
    blocks = [SummaryMemoryBlock(config) for _ in range(config.num_layers)]
    return CausalLM(config, blocks)
