from __future__ import annotations

from flower.config import ModelConfig
from flower.models.base import CausalLM
from flower.models.flow_attention import FlowSelfAttention
from flower.models.flow_memory import FlowMemoryBlock


def build_fa_fm_model(config: ModelConfig) -> CausalLM:
    blocks = []
    for _ in range(config.num_layers):
        block = FlowMemoryBlock(config)
        block.local = FlowSelfAttention(config, config.local_window)
        blocks.append(block)
    return CausalLM(config, blocks)
