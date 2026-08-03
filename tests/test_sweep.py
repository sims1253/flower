from __future__ import annotations

import json
from pathlib import Path

import yaml

from flower.sweep import deep_merge, expand_seed_variants, load_sweep, parse_seeds, run_sweep, select_variants


def test_deep_merge_preserves_defaults_and_overrides_nested_values() -> None:
    base = {"model": {"variant": "vanilla_local", "d_model": 64}, "training": {"steps": 10}}
    override = {"model": {"variant": "summary_memory", "local_window": None}}

    merged = deep_merge(base, override)

    assert merged == {
        "model": {"variant": "summary_memory", "d_model": 64, "local_window": None},
        "training": {"steps": 10},
    }
    assert base["model"] == {"variant": "vanilla_local", "d_model": 64}


def test_load_sweep_expands_defaults_with_per_variant_overrides(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        yaml.safe_dump(
            {
                "sweep": {
                    "name": "unit",
                    "defaults": {
                        "data": {"dataset": "synthetic", "sequence_length": 16},
                        "training": {"steps": 5, "batch_size": 2},
                        "model": {"variant": "vanilla_local", "d_model": 32, "local_window": 8},
                    },
                    "variants": [
                        {"name": "a", "model": {"variant": "vanilla_full", "local_window": None}},
                        {"name": "b", "training": {"lr": 0.001}},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    name, variants = load_sweep(sweep_path)

    assert name == "unit"
    assert variants[0]["config"]["model"] == {"variant": "vanilla_full", "d_model": 32, "local_window": None}
    assert variants[1]["config"]["training"] == {"steps": 5, "batch_size": 2, "lr": 0.001}


def test_select_variants_honors_comma_list_and_limit() -> None:
    variants = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    assert select_variants(variants, "c,a", 1) == [{"name": "c"}]


def test_expand_seed_variants_uses_training_seeds_and_names_outputs() -> None:
    variants = [
        {
            "name": "hier_max_16",
            "config": {
                "training": {"seed": 0, "seeds": [0, 1, 2]},
                "model": {"variant": "summary_memory"},
            },
        }
    ]

    expanded = expand_seed_variants(variants)

    assert [v["name"] for v in expanded] == ["hier_max_16_seed0", "hier_max_16_seed1", "hier_max_16_seed2"]
    assert [v["config"]["training"]["seed"] for v in expanded] == [0, 1, 2]
    assert all(v["base_name"] == "hier_max_16" for v in expanded)


def test_parse_seeds_overrides_config_seed_list() -> None:
    variants = [{"name": "a", "config": {"training": {"seed": 0, "seeds": [0, 1, 2]}}}]

    expanded = expand_seed_variants(variants, parse_seeds("7,8"))

    assert [v["name"] for v in expanded] == ["a_seed7", "a_seed8"]
    assert [v["config"]["training"]["seed"] for v in expanded] == [7, 8]


def test_run_sweep_smoke_writes_per_variant_metrics_and_summary(tmp_path: Path) -> None:
    sweep_path = tmp_path / "smoke_sweep.yaml"
    output_dir = tmp_path / "runs"
    sweep_path.write_text(
        yaml.safe_dump(
            {
                "sweep": {
                    "name": "smoke",
                    "defaults": {
                        "data": {"dataset": "synthetic", "sequence_length": 16, "synthetic_vocab_size": 128},
                        "training": {"steps": 3, "batch_size": 2, "device": "cpu"},
                        "model": {
                            "variant": "vanilla_local",
                            "vocab_size": 128,
                            "d_model": 32,
                            "num_heads": 4,
                            "num_layers": 1,
                            "ffn_dim": 64,
                            "max_seq_len": 16,
                            "local_window": 8,
                        },
                    },
                    "variants": [
                        {"name": "vanilla_local", "model": {"variant": "vanilla_local"}},
                        {"name": "summary_mean", "model": {"variant": "summary_memory", "memory_aggregation": "mean"}},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    summary = run_sweep(
        [
            "--config",
            str(sweep_path),
            "--output-dir",
            str(output_dir),
            "--steps",
            "1",
            "--device",
            "cpu",
            "--limit",
            "1",
            "--smoke",
        ]
    )

    metrics_path = output_dir / "vanilla_local.metrics.json"
    summary_path = output_dir / "summary.json"
    assert summary["variant_count"] == 1
    assert summary_path.exists()
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    written_summary = json.loads(summary_path.read_text())
    assert metrics["variant"] == "vanilla_local"
    assert metrics["steps"] == 1
    assert written_summary["variants"][0]["metrics_json"] == str(metrics_path)
    assert (output_dir / "variants" / "vanilla_local" / "tensorboard").exists()


def test_run_sweep_smoke_expands_configured_seeds(tmp_path: Path) -> None:
    sweep_path = tmp_path / "smoke_sweep.yaml"
    output_dir = tmp_path / "runs"
    sweep_path.write_text(
        yaml.safe_dump(
            {
                "sweep": {
                    "name": "smoke",
                    "defaults": {
                        "data": {"dataset": "synthetic", "sequence_length": 16, "synthetic_vocab_size": 128},
                        "training": {"steps": 1, "batch_size": 2, "device": "cpu", "seeds": [0, 1]},
                        "model": {
                            "variant": "vanilla_local",
                            "vocab_size": 128,
                            "d_model": 32,
                            "num_heads": 4,
                            "num_layers": 1,
                            "ffn_dim": 64,
                            "max_seq_len": 16,
                            "local_window": 8,
                        },
                    },
                    "variants": [{"name": "vanilla_local", "model": {"variant": "vanilla_local"}}],
                }
            }
        ),
        encoding="utf-8",
    )

    summary = run_sweep(
        [
            "--config",
            str(sweep_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--limit",
            "1",
            "--smoke",
        ]
    )

    assert summary["variant_count"] == 2
    assert (output_dir / "vanilla_local_seed0.metrics.json").exists()
    assert (output_dir / "vanilla_local_seed1.metrics.json").exists()
