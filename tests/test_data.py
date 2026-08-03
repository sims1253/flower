from __future__ import annotations

import torch

from flower.config import DataConfig
from flower.data import token_batches


def test_fineweb_edu_accepts_underscore_and_hyphen_aliases(monkeypatch) -> None:
    calls: list[tuple[DataConfig, int, torch.device]] = []

    def fake_fineweb_token_stream(config: DataConfig, batch_size: int, device: torch.device):
        calls.append((config, batch_size, device))
        return iter(())

    monkeypatch.setattr("flower.data.fineweb_token_stream", fake_fineweb_token_stream)
    device = torch.device("cpu")

    assert token_batches(DataConfig(dataset="fineweb_edu"), 2, device) is not None
    assert token_batches(DataConfig(dataset="fineweb-edu"), 3, device) is not None

    assert [call[0].dataset for call in calls] == ["fineweb_edu", "fineweb-edu"]
    assert [call[1] for call in calls] == [2, 3]
    assert all(call[2] == device for call in calls)


def test_synthetic_dataset_still_produces_batches() -> None:
    batches = token_batches(DataConfig(dataset="synthetic", sequence_length=8), 2, torch.device("cpu"))

    batch = next(batches)

    assert batch.shape == (2, 8)
    assert batch.device.type == "cpu"


def test_mqar_dataset_embeds_query_answers() -> None:
    cfg = DataConfig(dataset="mqar", sequence_length=64, synthetic_vocab_size=128)
    batches = token_batches(cfg, 2, torch.device("cpu"), seed=123)

    batch, labels = next(batches)

    assert batch.shape == (2, 64)
    assert labels.shape == (2, 64)
    # The stream should not degenerate to constant/random-size rows; values stay
    # inside the configured synthetic vocab.
    assert int(batch.min()) >= 0
    assert int(batch.max()) < cfg.synthetic_vocab_size
    assert int((labels != -100).sum()) > 0
    assert torch.equal(batch[labels != -100], labels[labels != -100])


def test_synthetic_validation_stream_uses_separate_seed() -> None:
    from flower.data import validation_token_batches

    cfg = DataConfig(dataset="synthetic", sequence_length=8, validation_seed=999)
    train_batch = next(token_batches(cfg, 2, torch.device("cpu")))
    val_batch = next(validation_token_batches(cfg, 2, torch.device("cpu")))

    assert not torch.equal(train_batch, val_batch)


def test_fineweb_validation_requires_distinct_split() -> None:
    from flower.data import validation_token_batches

    cfg = DataConfig(dataset="fineweb-edu", split="train", validation_split="train")
    try:
        validation_token_batches(cfg, 2, torch.device("cpu"))
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("expected ValueError")
