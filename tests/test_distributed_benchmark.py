import json
from pathlib import Path

from flower.distributed_benchmark import resolve_find_unused_parameters, run_benchmark


def test_distributed_benchmark_cpu_smoke(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    metrics = run_benchmark(
        [
            "--config",
            "configs/vanilla_local.yaml",
            "--variant",
            "vanilla_local",
            "--per-gpu-batch-size",
            "1",
            "--steps",
            "1",
            "--warmup-steps",
            "0",
            "--device",
            "cpu",
            "--metrics-json",
            str(metrics_path),
        ]
    )
    assert metrics is not None
    assert metrics["world_size"] == 1
    assert metrics["per_gpu_batch_size"] == 1
    assert metrics["global_batch_size"] == 1
    assert metrics["tokens_per_sec"] > 0
    assert metrics["loss"] > 0
    assert metrics["find_unused_parameters"] is False
    assert json.loads(metrics_path.read_text())["variant"] == "vanilla_local"


def test_find_unused_parameters_auto_known_variants():
    assert resolve_find_unused_parameters("auto", "fa_sm") is True
    assert resolve_find_unused_parameters("auto", "fa_fm") is True
    assert resolve_find_unused_parameters("auto", "vanilla_local") is False
    assert resolve_find_unused_parameters("true", "vanilla_local") is True
    assert resolve_find_unused_parameters("false", "fa_sm") is False
