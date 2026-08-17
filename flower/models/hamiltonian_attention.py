"""Hamiltonian flow on Q/K (symplectic-by-construction attention).

This variant replaces the Euler flow in `flow_attention` with a SympNet-style
Hamiltonian flow (see flower.flows.hamiltonian). The motivation: an Euler
flow on Q/K accumulates volume distortion and energy drift across layers with
no architectural defence; a symplectic flow preserves phase-space volume
exactly and conserves a shadow Hamiltonian, giving a flow whose long-range
behaviour is bounded by construction.

What carries over from `flow_attention`:
  - QKV projection, dense causal/local attention, output projection.
  - The same flow-step count config knob.

Positional encoding: none. Unlike CausalSelfAttention, this variant (like
`flow_attention`, which it inherits the pipeline from) applies no RoPE, and
the Hamiltonian flow on Q/K is position-wise, so it is not a stand-in
either. Attention here is permutation-equivariant within the causal/local
window — a known gap inherited from `flow_attention`. Adding RoPE would be
a behaviour change, not a cleanup, so it is left as a documented gap.

What is different:
  - Q and K are passed through HamiltonianFlow / WalnutsHamiltonianFlow
    instead of EulerFlow.
  - head_dim must be even (q/p split). For sweep4/5 dims (head_dim=64) this
    is satisfied.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.flows.hamiltonian import HamiltonianFlow, WalnutsHamiltonianFlow
from flower.models.base import (
    CausalLM,
    TransformerBlock,
    _get_or_build_block_mask,
    _get_or_build_causal_bool_mask,
    _load_flex_attention,
)


def _build_flow(head_dim: int, config: ModelConfig) -> nn.Module:
    flow_steps = max(1, config.flow_steps)
    if config.hamiltonian_walnuts:
        return WalnutsHamiltonianFlow(
            head_dim,
            steps=flow_steps,
            mass=config.hamiltonian_mass,
            energy_threshold=config.hamiltonian_energy_threshold,
            max_subdivisions=config.hamiltonian_max_subdivisions,
        )
    return HamiltonianFlow(head_dim, steps=flow_steps, mass=config.hamiltonian_mass)


class HamiltonianSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, local_window: int | None = None) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        head_dim = config.d_model // config.num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"hamiltonian_attention requires even head_dim for (q, p) split, got {head_dim}"
            )
        self.num_heads = config.num_heads
        self.head_dim = head_dim
        self.local_window = local_window
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)
        self.q_flow = _build_flow(head_dim, config)
        self.k_flow = self.q_flow if config.flow_shared else _build_flow(head_dim, config)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.noise_std = config.noise_std
        # S1 (FlexAttention) opt-in, mirroring CausalSelfAttention.
        self.use_flex = bool(getattr(config, "flex_attention", False))
        self._cached_block_mask = None
        self._cached_seq_len = 0
        self._cached_window = None
        # Dense-path bool mask cache (see _get_causal_bool_mask).
        self._cached_attn_mask = None
        self._cached_mask_seq_len = 0
        self._cached_mask_window = None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _get_block_mask(self, seq_len: int, device: torch.device):
        # Delegates to the shared, compile-safe cache logic (base.py).
        return _get_or_build_block_mask(self, seq_len, device)

    def _get_causal_bool_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Delegates to the shared, compile-safe cache logic (base.py).
        return _get_or_build_causal_bool_mask(self, seq_len, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        if self.training and self.noise_std:
            q = q + torch.randn_like(q) * self.noise_std
            k = k + torch.randn_like(k) * self.noise_std
        q = self.q_flow(q)
        k = self.k_flow(k)
        if self.use_flex:
            flex_attention, _ = _load_flex_attention()
            block_mask = self._get_block_mask(x.shape[1], x.device)
            out = flex_attention(q, k, v, block_mask=block_mask)
        else:
            # Fused SDPA with the cached bool mask instead of the legacy dense
            # path, which materialized (B, H, T, T) scores — the OOM/spill-prone
            # path CausalSelfAttention._forward_sdpa documents abandoning.
            # Numerically equivalent; pinned by tests/test_sdpa_attention_parity.py.
            mask = self._get_causal_bool_mask(x.shape[1], x.device).view(1, 1, x.shape[1], x.shape[1])
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(x.shape)
        return self.out(out)


def build_hamiltonian_attention_model(config: ModelConfig) -> CausalLM:
    blocks = [
        TransformerBlock(config, HamiltonianSelfAttention(config, config.local_window))
        for _ in range(config.num_layers)
    ]
    return CausalLM(config, blocks)
