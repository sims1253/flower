from __future__ import annotations

from flower.config import ModelConfig
from flower.models.base import CausalLM
from flower.models.flow_attention import FlowSelfAttention
from flower.models.flow_memory import FlowMemoryBlock


def build_fa_fm_model(config: ModelConfig) -> CausalLM:
    """fa_fm = FlowMemoryBlock with FlowSelfAttention as the local attention.

    causal_memory=True is honoured with no changes here: the write causality
    comes from FlowMemoryBlock._update_memory_causal and the read dispatches
    on the (B, T, S, D) memory state; FlowSelfAttention is a masked local
    causal attention in both modes."""
    blocks = []
    for _ in range(config.num_layers):
        block = FlowMemoryBlock(config)
        block.local = FlowSelfAttention(config, config.local_window)
        blocks.append(block)
    return CausalLM(config, blocks)
