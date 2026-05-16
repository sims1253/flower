from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    variant: str = "vanilla_local"
    vocab_size: int = 50257
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 1024
    max_seq_len: int = 512
    local_window: int | None = 64
    # Sweep 4: RoPE base frequency. Keep 10000 for the Phase 1 bake-off; larger
    # values can support post-training long-context extension via YaRN/PI.
    rope_base: float = 10000.0
    memory_slots: int = 8
    flow_steps: int = 3
    flow_shared: bool = True
    flow_mode: str = "euler"  # euler, direct, shortcut
    flow_step_size: float = 1.0
    memory_aggregation: str = "mean"  # sum, mean, max, attention, orthogonal
    summary_style: str = "deepsets"  # deepsets, perceiver
    hierarchical_memory: bool = False
    short_memory_slots: int = 4
    memory_kernel_bias: str = "none"  # none, positional, rbf
    memory_update_frequency: int = 1
    deep_shallow_flow: bool = False
    noise_std: float = 0.0
    dropout: float = 0.0
    # Sweep 2 additions
    loop_count: int = 1  # A2: loop blocks K times with shared memory state across loops
    local_attn_kernel_bias: str = "none"  # none, rbf — additive RBF bias on local self-attention
    local_attn_rbf_scale: float = 4.0  # initial scale for local-attention RBF (positions are 0..1 normalised)
    phase_memory_dim: int = 64  # A5: per-slot complex matrix is (phase_memory_dim x phase_memory_dim)
    orthogonal_eps: float = 1e-6  # A1: numerical floor for orthogonal projection
    num_memory_banks: int = 1  # I4: when >1, partition memory into N independent banks with a learned router
    bank_router_temperature: float = 1.0  # I4: softmax temperature for bank routing
    # Sweep 4: hashed n-gram residual memory. ngrams are hashed into a learned
    # table and added to the hidden state inside each block.
    engram_table_size: int = 8192
    engram_ngram_min: int = 2
    engram_ngram_max: int = 3
    engram_scale: float = 0.1
    # Sweep 4 (flow_meanflow): MeanFlow average-velocity parameterisation +
    # optional OT-CFM batch coupling. Loss weight scales the auxiliary regression
    # loss vs. the main cross-entropy. OT-CFM is off by default because it adds
    # O(B^2) coupling work; enable it explicitly for the OT-CFM ablation.
    meanflow_loss_weight: float = 0.1
    meanflow_ot_cfm: bool = False
    meanflow_ot_epsilon: float = 0.05
    meanflow_ot_iters: int = 20
    # Sweep 4 (flow_pma): number of coupling layers in the per-slot transport flow.
    flow_pma_layers: int = 2


@dataclass
class DataConfig:
    dataset: str = "synthetic"
    tokenizer: str = "gpt2"
    split: str = "train"
    sequence_length: int = 512
    streaming: bool = True
    text_field: str = "text"
    synthetic_vocab_size: int = 256
    validation_split: str | None = None
    validation_seed: int = 4321


@dataclass
class TrainingConfig:
    batch_size: int = 32
    steps: int = 30_000
    lr: float = 3e-4
    warmup_steps: int = 1000
    lr_schedule: str = "linear_warmup"  # constant, linear_warmup
    grad_clip: float = 1.0
    eval_interval: int = 500
    validation_interval: int = 0
    validation_steps: int = 0
    checkpoint_interval: int = 5000
    save_checkpoints: bool = True  # set false to skip torch.save during sweeps and save disk
    output_dir: str = "runs"
    metrics_json: str | None = None
    log_backend: str = "tensorboard"
    device: str = "auto"
    seed: int = 0
    seeds: tuple[int, ...] = (0, 1, 2)
    composite_eval: bool = False
    composite_eval_interval: int = 0
    composite_eval_json: str | None = None
    find_unused_parameters: bool | None = None
    # Sweep 2: optimizer choice. "adamw" = legacy default. "muon" = Muon for 2D weights + AdamW for 1D/embeddings/heads.
    optimizer: str = "adamw"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    # I8: optional faster LR for memory-bank parameters (Carmack's plasticity argument
    # — backbone learns slowly, memory adapts quickly). 0 disables; otherwise overrides
    # `lr` for params whose qualified name matches `memory_param_patterns`.
    memory_lr: float = 0.0
    memory_param_patterns: tuple[str, ...] = (
        "mem_read",
        "mem_mlp",
        "update",
        "perceiver",
        "agg_query",
        "short_project",
        "slot_bias",
        "rbf_scale",
        "proj_key",
        "proj_val",
        "proj_query",
        "proj_back",
        "decay",
        "cond",
        "flow",
    )


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _merge_dataclass(cls: type, values: dict[str, Any]) -> Any:
    base = cls()
    for key, value in values.items():
        if not hasattr(base, key):
            raise ValueError(f"Unknown config field {cls.__name__}.{key}")
        setattr(base, key, value)
    return base


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError("Config file must contain a mapping")
            raw.update(loaded)
    overrides = overrides or {}
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)

    return ExperimentConfig(
        model=_merge_dataclass(ModelConfig, raw.get("model", {})),
        data=_merge_dataclass(DataConfig, raw.get("data", {})),
        training=_merge_dataclass(TrainingConfig, raw.get("training", {})),
    )
