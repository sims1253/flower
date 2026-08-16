from __future__ import annotations

import json
from pathlib import Path

from flower.train import train


def test_tensorboard_logging_writes_event_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    metrics_json = tmp_path / "metrics.json"

    metrics = train(
        [
            "--variant",
            "summary_memory",
            "--smoke",
            "--steps",
            "1",
            "--device",
            "cpu",
            "--metrics-json",
            str(metrics_json),
            "--output-dir",
            str(output_dir),
            "--log-backend",
            "tensorboard",
        ]
    )

    event_files = list((output_dir / "tensorboard").glob("events.out.tfevents.*"))
    assert metrics["steps"] == 1
    assert metrics_json.exists()
    assert json.loads(metrics_json.read_text())["loss"] == metrics["loss"]
    assert event_files


def test_stability_scalars_are_logged(tmp_path: Path) -> None:
    """grad_norm / grad_norm_max / grad_clip_frac must reach TensorBoard.

    These are the signals that make training instability observable. Before they
    existed, `clip_grad_norm_`'s return value was computed and discarded, and
    `log_interval` was pinned to `eval_interval` (1000 in the production
    configs), so a 10k-step run recorded 11 loss points and no gradient
    information at all — you could not tell a stable run from an unstable one
    after the fact. Anything that silently drops these puts the pipeline back in
    that state, hence this test.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    output_dir = tmp_path / "run"
    train(
        [
            "--variant", "vanilla_local",
            "--smoke",
            "--steps", "3",
            "--device", "cpu",
            "--output-dir", str(output_dir),
            "--log-backend", "tensorboard",
        ]
    )

    ea = EventAccumulator(str(output_dir / "tensorboard"))
    ea.Reload()
    tags = ea.Tags()["scalars"]
    for tag in ("train/grad_norm", "train/grad_norm_max", "train/grad_clip_frac"):
        assert tag in tags, f"{tag} missing; stability would be unobservable"

    # The clip fraction is a fraction: it must be in [0, 1], not a raw count.
    for e in ea.Scalars("train/grad_clip_frac"):
        assert 0.0 <= e.value <= 1.0, f"grad_clip_frac={e.value} is not a fraction"
    # Norms are non-negative by construction; a negative one means the interval
    # accumulator was reset wrongly.
    for e in ea.Scalars("train/grad_norm"):
        assert e.value >= 0.0
