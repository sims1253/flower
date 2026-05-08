from __future__ import annotations

from flower.config import ModelConfig
from flower.models.base import CausalLM
from flower.models.flow_attention import build_flow_attention_model
from flower.models.flow_attn_flow import build_fa_fm_model
from flower.models.flow_attn_summary import build_fa_sm_model
from flower.models.flow_memory import build_flow_memory_model
from flower.models.linear_memory import build_linear_memory_model
from flower.models.summary_memory import build_summary_memory_model
from flower.models.vanilla import build_vanilla_model


def build_model(config: ModelConfig) -> CausalLM:
    variant = config.variant
    if variant == "vanilla_local":
        return build_vanilla_model(config, full_context=False)
    if variant == "vanilla_full":
        return build_vanilla_model(config, full_context=True)
    if variant == "linear_memory":
        return build_linear_memory_model(config)
    if variant == "summary_memory":
        return build_summary_memory_model(config)
    if variant == "flow_attention":
        return build_flow_attention_model(config)
    if variant == "flow_memory":
        return build_flow_memory_model(config)
    if variant == "fa_sm":
        return build_fa_sm_model(config)
    if variant == "fa_fm":
        return build_fa_fm_model(config)
    raise ValueError(f"Unknown model variant: {variant}")


__all__ = ["build_model"]
