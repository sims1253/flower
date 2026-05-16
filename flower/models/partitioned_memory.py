"""Partitioned Memory Banks (I4 — Keller's compute-near-data).

Splits the memory bank into N independent sub-banks, each with its own slots and
its own (key, value) projections. A tiny per-layer router scores each bank against
the layer's mean-pooled token representation and produces softmax weights w over
banks. Reads are weighted cross-attention across all bank slots; writes update each
bank with magnitude proportional to w[b].

Total slot count is held fixed vs the unpartitioned hierarchical_max baseline so the
comparison isolates "one bank vs many banks" rather than "more capacity."

For num_memory_banks=1 the architecture reduces to a hierarchical_max-equivalent
(modulo the routing scalar = 1.0).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward


class PartitionedMemoryRead(nn.Module):
    """Cross-attention over N bank-flattened slots, additively biased by router weights."""

    def __init__(self, config: ModelConfig, num_banks: int, slots_per_bank: int) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.num_banks = num_banks
        self.slots_per_bank = slots_per_bank
        self.q = nn.Linear(config.d_model, config.d_model)
        self.kv = nn.Linear(config.d_model, config.d_model * 2)
        self.out = nn.Linear(config.d_model, config.d_model)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, bank_log_weights: torch.Tensor) -> torch.Tensor:
        # memory: (B, N, S, D) → flatten to (B, N*S, D); bank_log_weights: (B, N).
        bsz, num_banks, slots_per_bank, dim = memory.shape
        mem_flat = memory.reshape(bsz, num_banks * slots_per_bank, dim)
        q = self._split(self.q(x))
        k, v = self.kv(mem_flat).chunk(2, dim=-1)
        k, v = self._split(k), self._split(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        # Additive bias: add log(w_b) to the slots belonging to bank b. This makes
        # the router weights act multiplicatively on attention probabilities.
        # bank_log_weights: (B, N) → (B, 1, 1, N) → tile across slots → (B, 1, 1, N*S).
        bias = bank_log_weights.unsqueeze(1).unsqueeze(1).repeat_interleave(slots_per_bank, dim=-1)
        # Already shape (B, 1, 1, N*S); broadcasts cleanly to (B, H, T, N*S).
        scores = scores + bias
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(x.shape)
        return self.out(out)


class PartitionedMemoryBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_banks = max(1, int(config.num_memory_banks))
        # Distribute slots evenly across banks. For hierarchical_max with 8+4=12 slots
        # and num_banks=4 → 3 slots per bank (we keep total fixed).
        total_slots = config.memory_slots + (config.short_memory_slots if config.hierarchical_memory else 0)
        if total_slots % self.num_banks != 0:
            raise ValueError(f"Total slots ({total_slots}) must be divisible by num_memory_banks ({self.num_banks})")
        self.slots_per_bank = total_slots // self.num_banks

        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = PartitionedMemoryRead(config, self.num_banks, self.slots_per_bank)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout)

        # Per-bank update MLPs. Sharing the same head saves params; one set covers all banks.
        self.token_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model)
        )
        self.mem_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model)
        )
        self.update = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model)
        )

        # I4 router: scores each bank from the mean-pooled token representation.
        self.router = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, self.num_banks),
        )
        self.router_temperature = float(config.bank_router_temperature)

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], self.num_banks, self.slots_per_bank, self.config.d_model)

    def _bank_weights(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (soft_weights, log_weights) of shape (B, N).

        Routing computed from per-batch mean-pooled token rep (one routing decision
        per layer per batch element — Keller "found parallelism" / Carmack BSP style:
        coarse routing decision lets you skip work in non-selected banks).
        """
        pooled = x.mean(dim=1)  # (B, D)
        logits = self.router(pooled) / max(self.router_temperature, 1e-6)
        log_weights = F.log_softmax(logits, dim=-1)
        return log_weights.exp(), log_weights

    def _update_memory(self, memory: torch.Tensor, x: torch.Tensor, bank_weights: torch.Tensor) -> torch.Tensor:
        # token_summary: per-bank max-pool aggregation, broadcast to slots_per_bank.
        token_summary = x.max(dim=1, keepdim=True).values  # (B, 1, D)
        # Compute candidate update for each bank (parameter-shared MLPs, but the
        # additive `combined = token_mlp(summary) + mem_mlp(slot)` differs per slot).
        # Build (B, N, S, D) candidate update.
        candidate = self.update(
            self.token_mlp(token_summary).unsqueeze(1)  # (B, 1, 1, D)
            + self.mem_mlp(memory)  # (B, N, S, D)
        ) / max(1, self.config.num_layers)
        # Scale per-bank update by router weight w_b — banks the router didn't
        # select get near-zero updates that step (compute-near-data).
        scale = bank_weights.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        return memory + scale * candidate

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory is None or memory.dim() != 4 or memory.shape[1] != self.num_banks:
            memory = self._initial_memory(x)
        x = x + self.local(self.ln1(x))
        bank_w, bank_logw = self._bank_weights(self.ln_mem(x))
        x = x + self.mem_read(self.ln_mem(x), memory, bank_logw)
        x = x + self.ff(self.ln2(x))
        if self.config.memory_update_frequency <= 1 or x.shape[1] % self.config.memory_update_frequency == 0:
            memory = self._update_memory(memory, x, bank_w)
        return x, memory


def build_partitioned_memory_model(config: ModelConfig) -> CausalLM:
    blocks = [PartitionedMemoryBlock(config) for _ in range(config.num_layers)]
    return CausalLM(config, blocks)
