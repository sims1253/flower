from __future__ import annotations

from flower.config import ModelConfig
from flower.models.base import (
    CausalLM,
    CausalSelfAttention,
    TransformerBlock,
    layer_attn_windows,
    layer_ffn_dims,
)


def build_vanilla_model(config: ModelConfig, full_context: bool = False) -> CausalLM:
    cfg = ModelConfig(**{**config.__dict__})
    if full_context:
        cfg.local_window = None
        cfg.attn_window_schedule = None
    ffn_dims = layer_ffn_dims(cfg)
    windows = layer_attn_windows(cfg)
    blocks = [
        TransformerBlock(
            cfg,
            attention=CausalSelfAttention(cfg, windows[i]),
            ffn_dim=ffn_dims[i],
        )
        for i in range(cfg.num_layers)
    ]
    return CausalLM(cfg, blocks)
