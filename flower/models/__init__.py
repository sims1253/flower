from __future__ import annotations

from flower.config import ModelConfig
from flower.models.base import CausalLM
from flower.models.bloom_memory import build_bloom_memory_model
from flower.models.engram_lite import build_engram_lite_model
from flower.models.fla_layer import build_fla_gdn_model
from flower.models.flow_attention import build_flow_attention_model
from flower.models.flow_attn_flow import build_fa_fm_model
from flower.models.flow_attn_summary import build_fa_sm_model
from flower.models.flow_meanflow import build_flow_meanflow_model
from flower.models.flow_memory import build_flow_memory_model
from flower.models.flow_ot_memory import build_flow_ot_memory_model
from flower.models.flow_pma import build_flow_pma_model
from flower.models.frequency_decay_memory import build_frequency_decay_memory_model
from flower.models.hamiltonian_attention import build_hamiltonian_attention_model
from flower.models.linear_memory import build_linear_memory_model
from flower.models.partitioned_memory import build_partitioned_memory_model
from flower.models.phase_memory import build_phase_memory_model
from flower.models.still_lm import build_still_model
from flower.models.summary_memory import build_summary_memory_model
from flower.models.surprise_memory import build_surprise_memory_model
from flower.models.titans_mac import build_titans_mac_model
from flower.models.vanilla import build_vanilla_model


# NOTE: the base causal-memory PR (#5) briefly guarded the four flow hybrids
# (flow_memory, flow_meanflow, flow_pma, fa_fm) behind
# CAUSAL_MEMORY_UNSUPPORTED_VARIANTS because their writes ignored the flag.
# This PR implements their causal writes, so every memory variant now honours
# causal_memory and the guard is removed in full (see CAUSAL_MEMORY_VARIANTS
# in tests/test_causal.py for the coverage matrix).


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
    if variant == "flow_pma":
        return build_flow_pma_model(config)
    if variant == "flow_attention":
        return build_flow_attention_model(config)
    if variant == "flow_memory":
        return build_flow_memory_model(config)
    if variant == "flow_meanflow":
        return build_flow_meanflow_model(config)
    if variant == "fa_sm":
        return build_fa_sm_model(config)
    if variant == "fa_fm":
        return build_fa_fm_model(config)
    if variant == "phase_memory":
        return build_phase_memory_model(config)
    if variant == "partitioned_memory":
        return build_partitioned_memory_model(config)
    if variant == "engram_lite":
        return build_engram_lite_model(config)
    if variant == "titans_mac":
        return build_titans_mac_model(config)
    if variant == "flow_ot_memory":
        return build_flow_ot_memory_model(config)
    if variant == "surprise_memory":
        return build_surprise_memory_model(config)
    if variant == "frequency_decay_memory":
        return build_frequency_decay_memory_model(config)
    if variant == "bloom_memory":
        return build_bloom_memory_model(config)
    if variant == "hamiltonian_attention":
        return build_hamiltonian_attention_model(config)
    if variant in {"fla_gdn", "fla_layer"}:
        return build_fla_gdn_model(config)
    if variant.startswith("still"):
        return build_still_model(config)
    raise ValueError(f"Unknown model variant: {variant}")


__all__ = ["build_model"]
