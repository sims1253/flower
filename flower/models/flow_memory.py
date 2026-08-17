from __future__ import annotations

import torch
from torch import nn

from flower.config import ModelConfig
from flower.flows.coupling import ConditionalCouplingFlow
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward
from flower.models.memory import MemoryRead, causal_chunked_map, causal_running_mean


class FlowMemoryBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = MemoryRead(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout, config=config)
        flat_dim = config.memory_slots * config.d_model
        if flat_dim % 2:
            flat_dim += 1
        self.flat_dim = flat_dim
        self.cond = nn.Linear(config.d_model, config.d_model)
        # hidden_dim must scale modestly with d_model (NOT with flat_dim = slots*d_model).
        # History: original default `hidden_dim = dim*2` gave 504M at d=256 (broken).
        # First fix: `hidden_dim = 2*d_model` gave 49M at d=256 (fine) but 184M at d=384.
        # Second fix: `hidden_dim = d_model//2` keeps fa_fm in the 40-70M regime alongside
        # the other variants across the 256d-384d range.
        flow_hidden = max(64, config.d_model // 2)
        self.flow = ConditionalCouplingFlow(flat_dim, config.d_model, layers=2, hidden_dim=flow_hidden)

    def _flat_memory(self, memory: torch.Tensor) -> torch.Tensor:
        # Shape-agnostic: flattens the trailing (S, D) axes (or just D) into
        # one feature axis. Legacy 3-D (B, S, D) input yields exactly the
        # original (B, flat) result; causal 4-D (B, T, S, D) yields (B, T, flat).
        flat = memory.reshape(*memory.shape[:-2], -1) if memory.dim() >= 3 else memory.reshape(memory.shape[0], -1)
        if flat.shape[-1] < self.flat_dim:
            flat = torch.nn.functional.pad(flat, (0, self.flat_dim - flat.shape[-1]))
        return flat

    def _unflat_memory(self, flat: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        size = self.config.memory_slots * self.config.d_model
        return flat[..., :size].reshape_as(memory)

    def _initial_memory_causal(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, S, D) per-position memory state for causal_memory=True."""
        return x.new_zeros(x.shape[0], x.shape[1], self.config.memory_slots, self.config.d_model)

    def _update_memory_causal(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """causal_memory=True write: memory is a per-position state (B, T, S, D).

        Same coupling flow and conditioning projection as the legacy write
        (identical parameters, no new ones), but the condition at t is the
        prefix mean of tokens <= t (legacy: the whole-window mean, which
        includes future tokens) and the flow transports the slot state at t
        only. ``ConditionalCouplingFlow`` operates on the last dim, so it
        applies to the (B, T, flat) per-position batch unchanged; the
        application is chunked over T to keep the coupling nets' fp32
        intermediates bounded (see ``causal_chunked_map``).
        """
        cond = self.cond(causal_running_mean(x))  # (B, T, D)
        flat = self._flat_memory(memory)  # (B, T, flat_dim)
        new_flat = causal_chunked_map(self.flow, flat, cond)
        return self._unflat_memory(new_flat, memory)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory is None:
            if self.config.causal_memory:
                memory = self._initial_memory_causal(x)
            else:
                memory = x.new_zeros(x.shape[0], self.config.memory_slots, self.config.d_model)
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))
        if self.config.causal_memory:
            memory = self._update_memory_causal(memory, x)
        else:
            cond = self.cond(x.mean(dim=1))
            memory = self._unflat_memory(self.flow(self._flat_memory(memory), cond), memory)
        return x, memory

    def inverse_update(self, new_memory: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        cond = self.cond(cond_tokens.mean(dim=1))
        flat = self.flow.inverse(self._flat_memory(new_memory), cond)
        return self._unflat_memory(flat, new_memory)


def build_flow_memory_model(config: ModelConfig) -> CausalLM:
    blocks = [FlowMemoryBlock(config) for _ in range(config.num_layers)]
    return CausalLM(config, blocks)
