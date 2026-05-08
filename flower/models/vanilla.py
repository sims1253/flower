from __future__ import annotations

from flower.config import ModelConfig
from flower.models.base import CausalLM, TransformerBlock


def build_vanilla_model(config: ModelConfig, full_context: bool = False) -> CausalLM:
    cfg = ModelConfig(**{**config.__dict__})
    if full_context:
        cfg.local_window = None
    blocks = [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
    return CausalLM(cfg, blocks)
