"""Wrapper around `fla.layers.GatedDeltaNet` from flash-linear-attention.

Background: Gated DeltaNet (Yang et al., 2024, arXiv:2412.06464) augments the delta
rule with input-dependent gating. The FLA implementation uses Triton kernels for
chunked recurrence and is the same code path that powers Qwen3.5's linear-attention
variants. Drop-in here so the Phase 1 bake-off compares to a real, published
architecture rather than a hand-rolled stand-in.

This wrapper:
- Routes Flower's (B, T, D) tensors through the FLA layer (which uses the same
  layout) and unwraps the tuple return value.
- Reuses Flower's `FeedForward` + LayerNorm pre-norm structure so the rest of the
  block matches the other Phase 1 candidates.
- Does NOT carry memory across blocks (the FLA layer maintains its recurrence state
  internally during the forward pass over the time dimension).
"""

from __future__ import annotations

import torch
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, FeedForward


def _resolve_head_dim(config: ModelConfig) -> int:
    if config.d_model % config.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads for FLA layer")
    return config.d_model // config.num_heads


class FLAGatedDeltaBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # Lazy import: fla pulls in Triton at import time, which we only want to pay
        # for if this variant is actually selected.
        from fla.layers import GatedDeltaNet

        head_dim = _resolve_head_dim(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        # FLA GatedDeltaNet expects hidden_size and head_dim; num_heads is derived
        # internally as hidden_size // head_dim when num_heads is omitted, but we pass
        # it explicitly so a mismatch fails loudly.
        self.attn = GatedDeltaNet(
            hidden_size=config.d_model,
            head_dim=head_dim,
            num_heads=config.num_heads,
            mode="chunk",
            use_gate=True,
            use_short_conv=True,
            conv_size=4,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout, config=config)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # FLA returns (hidden_states, attn_weights_or_None, past_key_values_or_None).
        h = self.ln1(x)
        attn_out, _, _ = self.attn(hidden_states=h)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, memory


def build_fla_gdn_model(config: ModelConfig) -> CausalLM:
    blocks = [FLAGatedDeltaBlock(config) for _ in range(config.num_layers)]
    return CausalLM(config, blocks)
