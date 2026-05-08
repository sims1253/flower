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
