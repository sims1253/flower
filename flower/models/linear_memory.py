from __future__ import annotations

import torch
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward
from flower.models.memory import MemoryRead, causal_running_mean


class LinearMemoryBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = MemoryRead(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout, config=config)
        self.write = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory is None:
            if self.config.causal_memory:
                # Per-position memory state (B, T, S, D); the read at t
                # consumes the state at t (MemoryRead dispatches on dim).
                memory = x.new_zeros(x.shape[0], x.shape[1], self.config.memory_slots, self.config.d_model)
            else:
                memory = x.new_zeros(x.shape[0], self.config.memory_slots, self.config.d_model)
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))
        if self.config.causal_memory:
            # Legacy writes the mean over the WHOLE window (future tokens
            # included) into every slot. Causal form: prefix mean at t, so
            # slot state at t sees tokens <= t only. Same `write` projection.
            summary = causal_running_mean(x).unsqueeze(2)  # (B, T, 1, D)
        else:
            summary = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        memory = memory + self.write(summary).expand_as(memory)
        return x, memory


def build_linear_memory_model(config: ModelConfig) -> CausalLM:
    blocks = [LinearMemoryBlock(config) for _ in range(config.num_layers)]
    return CausalLM(config, blocks)
