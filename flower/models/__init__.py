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


# ---------------------------------------------------------------------------
# causal_memory support map (see ModelConfig.causal_memory).
#
# Variants listed here do NOT implement the causal memory write yet: their
# blocks ignore config.causal_memory entirely, so a user setting it to True
# would silently train the legacy (whole-window, future-leaking) write while
# believing they run causal. Fail loudly at construction instead.
#
# fa_sm is deliberately NOT listed: it builds SummaryMemoryBlocks, which are
# fully causal under the flag (measured last-token leak exactly 0.0).
#
# The causal-flow-hybrids follow-up PR (#12, fix/causal-flow-hybrids) fixes
# the four flow hybrids below and removes them from this set — keep the guard
# as a set + single check so that update is a one-line edit.
# ---------------------------------------------------------------------------
CAUSAL_MEMORY_UNSUPPORTED_VARIANTS = frozenset({"flow_memory", "flow_meanflow", "flow_pma", "fa_fm"})


def build_model(config: ModelConfig) -> CausalLM:
    variant = config.variant
    if config.causal_memory and variant in CAUSAL_MEMORY_UNSUPPORTED_VARIANTS:
        raise ValueError(
            f"variant {variant!r} does not support causal_memory=True yet: its memory "
            f"write still aggregates the whole window (future tokens included), so the "
            f"flag would be silently ignored. Run it with causal_memory=False (the "
            f"legacy behaviour), or use a supported variant. {variant!r} is fixed by "
            f"the causal-flow-hybrids follow-up PR (#12, fix/causal-flow-hybrids)."
        )
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
