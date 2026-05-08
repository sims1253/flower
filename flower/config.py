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
    memory_slots: int = 8
    flow_steps: int = 3
    flow_shared: bool = True
    flow_mode: str = "euler"  # euler, direct, shortcut
    flow_step_size: float = 1.0
    memory_aggregation: str = "mean"  # sum, mean, max, attention
    summary_style: str = "deepsets"  # deepsets, perceiver
    hierarchical_memory: bool = False
    short_memory_slots: int = 4
    memory_kernel_bias: str = "none"  # none, positional, rbf
    memory_update_frequency: int = 1
    deep_shallow_flow: bool = False
    noise_std: float = 0.0
    dropout: float = 0.0


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
    grad_clip: float = 1.0
    eval_interval: int = 500
    validation_interval: int = 0
    validation_steps: int = 0
    checkpoint_interval: int = 5000
    output_dir: str = "runs"
    metrics_json: str | None = None
    log_backend: str = "tensorboard"
    device: str = "auto"
    find_unused_parameters: bool | None = None


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
