from __future__ import annotations

import torch
from torch import nn

from flower.config import ModelConfig
from flower.flows.cnf import EulerFlow
from flower.models.base import (
    CausalLM,
    TransformerBlock,
    _load_flex_attention,
    causal_mask,
    make_causal_local_block_mask,
    scaled_dot_attention,
)


class FlowSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, local_window: int | None = None) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.local_window = local_window
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)
        flow_steps = 1 if config.deep_shallow_flow and config.flow_steps > 1 else config.flow_steps
        self.q_flow = EulerFlow(self.head_dim, flow_steps, mode=config.flow_mode, step_size=config.flow_step_size)
        self.k_flow = (
            self.q_flow
            if config.flow_shared
            else EulerFlow(self.head_dim, flow_steps, mode=config.flow_mode, step_size=config.flow_step_size)
        )
        self.out = nn.Linear(config.d_model, config.d_model)
        self.q_flow.noise_std = config.noise_std
        # S1 (FlexAttention) opt-in, mirroring CausalSelfAttention.
        self.use_flex = bool(getattr(config, "flex_attention", False))
        self._cached_block_mask = None
        self._cached_seq_len = 0
        self._cached_window = None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _get_block_mask(self, seq_len: int, device: torch.device):
        window = self.local_window
        if (
            self._cached_block_mask is None
            or seq_len != self._cached_seq_len
            or window != self._cached_window
        ):
            self._cached_block_mask = make_causal_local_block_mask(window, seq_len, device)
            self._cached_seq_len = seq_len
            self._cached_window = window
        return self._cached_block_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        if self.training and self.q_flow.training:
            noise_std = getattr(self.q_flow, "noise_std", 0.0)
            if noise_std:
                q = q + torch.randn_like(q) * noise_std
                k = k + torch.randn_like(k) * noise_std
        q = self.q_flow(q)
        k = self.k_flow(k)
        if self.use_flex:
            flex_attention, _ = _load_flex_attention()
            block_mask = self._get_block_mask(x.shape[1], x.device)
            out = flex_attention(q, k, v, block_mask=block_mask)
        else:
            mask = causal_mask(x.shape[1], x.device, self.local_window).view(1, 1, x.shape[1], x.shape[1])
            out = scaled_dot_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(x.shape)
        return self.out(out)


def build_flow_attention_model(config: ModelConfig) -> CausalLM:
    blocks = [
        TransformerBlock(config, FlowSelfAttention(config, config.local_window)) for _ in range(config.num_layers)
    ]
    return CausalLM(config, blocks)
