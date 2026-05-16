from flower.train import train


def test_smoke_training_step_runs():
    metrics = train(["--variant", "summary_memory", "--smoke", "--steps", "1", "--device", "cpu"])
    assert metrics["steps"] == 1
    assert metrics["loss"] > 0
    assert metrics["tokens_per_sec"] > 0


def test_smoke_training_writes_validation_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics = train(
        [
            "--variant",
            "vanilla_local",
            "--smoke",
            "--steps",
            "1",
            "--device",
            "cpu",
            "--validation-steps",
            "1",
            "--metrics-json",
            str(metrics_path),
            "--log-backend",
            "none",
        ]
    )

    assert metrics["train_loss"] > 0
    assert metrics["val_loss"] > 0
    assert metrics["val_perplexity"] > 0
    assert metrics["val_tokens"] == 64
    assert metrics_path.exists()
