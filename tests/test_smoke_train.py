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
    assert metrics["val_tokens"] == 62
    assert metrics_path.exists()


# ---------------------------------------------------------------------------
# Checkpoint resume (train.py resume region). --smoke disables checkpointing,
# so these run the tiny smoke-sized model through a config file instead.
# ---------------------------------------------------------------------------


def _write_train_config(tmp_path, steps: int, accum: int = 2, name: str = "cfg.yaml") -> str:
    import yaml

    cfg = {
        "model": {
            "d_model": 32,
            "num_heads": 4,
            "num_layers": 1,
            "ffn_dim": 64,
            "max_seq_len": 32,
            "local_window": 8,
        },
        "data": {
            "dataset": "synthetic",
            "sequence_length": 32,
            "synthetic_vocab_size": 256,
            "eval_seq_len": 32,
        },
        "training": {
            "steps": steps,
            "batch_size": 2,
            "gradient_accumulation_steps": accum,
            "checkpoint_interval": 1,
            "save_checkpoints": True,
            "log_backend": "none",
            "device": "cpu",
            "output_dir": str(tmp_path),
            "eval_interval": 1,
        },
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_train_resume_replays_data_cursor_and_continues(tmp_path, capsys):
    import torch

    # Run 1: two steps, one checkpoint per step (last 2 kept).
    train(["--config", _write_train_config(tmp_path, steps=2)])
    ckpts = sorted(tmp_path.glob("vanilla_local_step*.pt"), key=lambda p: int(p.stem.split("step")[-1]))
    assert [p.stem.split("step")[-1] for p in ckpts] == ["1", "2"]

    # The checkpoint carries the exact data cursor: skip(0) + step * accum.
    payload = torch.load(ckpts[-1], weights_only=True)
    assert payload["data_batches_consumed"] == 4  # step 2 * accum 2

    # Run 2: resumes from the step-2 checkpoint and trains only step 3.
    train(["--config", _write_train_config(tmp_path, steps=3, name="cfg2.yaml")])
    out = capsys.readouterr().out
    assert "resuming from step 3 (checkpoint at step 2)" in out
    assert "[resume] replaying 4 already-consumed batches" in out
    # Progress logging fires at least at the end of the replay.
    assert "data replay 4/4 (100%" in out

    # The step-3 checkpoint composes the cursor: skip(4) + 1 step * accum 2.
    ckpts = sorted(tmp_path.glob("vanilla_local_step*.pt"), key=lambda p: int(p.stem.split("step")[-1]))
    assert ckpts[-1].stem.split("step")[-1] == "3"
    payload = torch.load(ckpts[-1], weights_only=True)
    assert payload["data_batches_consumed"] == 6


def test_train_resume_rejects_optimizer_count_mismatch(tmp_path):
    import torch

    train(["--config", _write_train_config(tmp_path, steps=1)])
    ckpt = tmp_path / "vanilla_local_step1.pt"
    payload = torch.load(ckpt, weights_only=True)
    payload["optimizers"] = []  # e.g. the optimizer config changed between runs
    torch.save(payload, ckpt)

    import pytest

    with pytest.raises(RuntimeError, match="optimizer"):
        train(["--config", _write_train_config(tmp_path, steps=2, name="cfg2.yaml")])
