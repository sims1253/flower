import json
from pathlib import Path

import torch

from flower.config import ModelConfig, load_config
from flower.eval import evaluate
from flower.train import train
from flower.models import build_model


ROOT = Path(__file__).resolve().parents[1]


def tiny_config(**kwargs):
    values = dict(variant="fa_sm", vocab_size=128, d_model=32, num_heads=4, num_layers=1, ffn_dim=64, max_seq_len=16, local_window=4, memory_slots=4, flow_steps=1)
    values.update(kwargs)
    return ModelConfig(**values)


def test_config_constructs_new_research_options():
    cfg = tiny_config(flow_mode="shortcut", memory_aggregation="attention", summary_style="perceiver", hierarchical_memory=True, memory_kernel_bias="rbf", memory_update_frequency=2, noise_std=0.01)
    model = build_model(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(tokens, labels=tokens)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert out["loss"].ndim == 0


def test_aggregation_and_flow_modes_smoke():
    for aggregation in ["sum", "mean", "max", "attention"]:
        cfg = tiny_config(variant="summary_memory", memory_aggregation=aggregation, hierarchical_memory=True)
        out = build_model(cfg)(torch.randint(0, cfg.vocab_size, (1, 8)))
        assert out["logits"].shape == (1, 8, cfg.vocab_size)
    for mode in ["direct", "shortcut"]:
        cfg = tiny_config(variant="flow_attention", flow_mode=mode, flow_step_size=0.5)
        out = build_model(cfg)(torch.randint(0, cfg.vocab_size, (1, 8)))
        assert out["logits"].shape == (1, 8, cfg.vocab_size)


def test_eval_writes_metrics_json(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics = evaluate(["--variant", "flow_attention", "--smoke", "--batches", "1", "--device", "cpu", "--metrics-json", str(metrics_path)])
    written = json.loads(metrics_path.read_text())
    assert written["variant"] == "flow_attention"
    assert written["gpu_memory_allocated"] == 0
    assert metrics["parameter_count"] == written["parameter_count"]


def test_train_writes_metrics_json(tmp_path):
    metrics_path = tmp_path / "train_metrics.json"
    metrics = train(["--variant", "fa_sm", "--smoke", "--steps", "1", "--device", "cpu", "--metrics-json", str(metrics_path)])
    written = json.loads(metrics_path.read_text())
    assert written["variant"] == "fa_sm"
    assert written["steps"] == 1
    assert written["gpu_memory_allocated"] == 0
    assert metrics["parameter_count"] == written["parameter_count"]


def test_sweep_config_loads():
    raw = load_config(ROOT / "configs" / "flow_attention.yaml")
    assert raw.model.variant == "flow_attention"
