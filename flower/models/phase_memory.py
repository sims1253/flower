"""Phase-Associative Memory (Plate 1995-style holographic binding).

Each memory slot stores a complex vector m_s in C^P that holds a *superposition*
of (key, value) bindings created via element-wise complex multiplication. To read
a value associated with a query q, we unbind: v_hat = m_s * conj(q). On unit-norm
complex vectors, k * conj(k) ≈ 1 so the bound value is approximately recovered.

This is mathematically distinct from attention-based retrieval. Capacity scales
roughly as O(P / log P) reliable bindings per slot — an information-theoretic
property of HRR-style codes (Plate 2003). Risky at small scale but cheap to test.

PyTorch supports complex64 natively; we keep complex ops in float32 to avoid
autocast issues. The block layout otherwise mirrors SummaryMemoryBlock so the
local-attention path / FFN are interchangeable with other variants.
"""

from __future__ import annotations

import torch
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward


def _to_unit_complex(x: torch.Tensor) -> torch.Tensor:
    """Split a real (..., 2P) tensor into a unit-norm complex (..., P) tensor.

    Per-element normalisation (each complex scalar lies on the unit circle) gives
    the property k * conj(k) = 1 needed for HRR-style unbinding.
    """
    re, im = x.chunk(2, dim=-1)
    re = re.float()
    im = im.float()
    z = torch.complex(re, im)
    mag = z.abs().clamp(min=1e-6)
    return z / mag


def _from_complex(z: torch.Tensor) -> torch.Tensor:
    """Inverse of `_to_unit_complex`'s shape change: complex (..., P) → real (..., 2P)."""
    return torch.cat([z.real, z.imag], dim=-1)


class PhaseAssociativeBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.P = int(config.phase_memory_dim)
        self.S = int(config.memory_slots)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout)
        # Per-layer projections from token reps to (per-slot) complex key/value pairs.
        # We use one linear that emits S * 2P then reshape to (S, 2P) → S complex vectors of dim P.
        self.proj_key = nn.Linear(config.d_model, self.S * 2 * self.P)
        self.proj_val = nn.Linear(config.d_model, self.S * 2 * self.P)
        # Read query: per-token complex query of dim P (one query, attends to all slots).
        self.proj_query = nn.Linear(config.d_model, 2 * self.P)
        # Map retrieved complex vector back into the model's real d_model space.
        self.proj_back = nn.Linear(2 * self.P, config.d_model)
        # Learnable decay toward the running superposition (sigmoid'd in forward).
        self.decay = nn.Parameter(torch.tensor(2.0))  # sigmoid(2) ≈ 0.88

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.S, self.P, dtype=torch.complex64, device=x.device)

    def _aggregate_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce (B, T, D) → (B, D) via max-pool over time. Same signal as the Sweep 1 winner."""
        return x.max(dim=1).values

    def _write(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pooled = self._aggregate_tokens(x)  # (B, D)
        bsz = pooled.shape[0]
        # (B, S, 2P) → S complex unit vectors of dim P per batch element.
        keys = _to_unit_complex(self.proj_key(pooled).view(bsz, self.S, 2 * self.P))
        vals = _to_unit_complex(self.proj_val(pooled).view(bsz, self.S, 2 * self.P))
        # Hadamard binding: each slot accumulates one (key * val) per layer.
        binding = keys * vals  # (B, S, P) complex
        decay = torch.sigmoid(self.decay).float()
        decay_c = torch.complex(decay, torch.zeros_like(decay))
        return decay_c * memory + (1.0 - decay_c) * binding

    def _read(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) — produce a complex query per token.
        bsz, T, _ = x.shape
        q = _to_unit_complex(self.proj_query(x))  # (B, T, P) complex
        # Unbind each slot with the per-token query: m_s * conj(q) → (B, T, S, P).
        unbinds = memory.unsqueeze(1) * torch.conj(q).unsqueeze(2)
        # Aggregate over slots (mean): one combined retrieved vector per token.
        retrieved = unbinds.mean(dim=2)  # (B, T, P) complex
        # Back to real d_model space.
        real = _from_complex(retrieved).to(x.dtype)
        return self.proj_back(real)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory is None or not torch.is_complex(memory):
            memory = self._initial_memory(x)
        x = x + self.local(self.ln1(x))
        x = x + self._read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))
        memory = self._write(memory, x)
        return x, memory


def build_phase_memory_model(config: ModelConfig) -> CausalLM:
    blocks = [PhaseAssociativeBlock(config) for _ in range(config.num_layers)]
    return CausalLM(config, blocks)
