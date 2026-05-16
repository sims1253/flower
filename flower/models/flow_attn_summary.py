from __future__ import annotations

from flower.config import ModelConfig
from flower.flows.cnf import EulerFlow
from flower.models.base import CausalLM
from flower.models.flow_attention import FlowSelfAttention
from flower.models.memory import MemoryRead
from flower.models.summary_memory import SummaryMemoryBlock


def build_fa_sm_model(config: ModelConfig) -> CausalLM:
    blocks = []
    for _ in range(config.num_layers):
        block = SummaryMemoryBlock(
            config,
            MemoryRead(
                config,
                EulerFlow(
                    config.d_model // config.num_heads,
                    config.flow_steps,
                    mode=config.flow_mode,
                    step_size=config.flow_step_size,
                ),
            ),
        )
        block.local = FlowSelfAttention(config, config.local_window)
        blocks.append(block)
    return CausalLM(config, blocks)
