from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flower.config import DataConfig
from flower.data import token_batches


def test_fineweb_edu_accepts_underscore_and_hyphen_aliases(monkeypatch) -> None:
    calls: list[tuple[DataConfig, int, torch.device]] = []

    def fake_fineweb_token_stream(
        config: DataConfig,
        batch_size: int,
        device: torch.device,
        *,
        num_workers: int | None = None,
        prefetch_factor: int | None = None,
    ):
        calls.append((config, batch_size, device, num_workers, prefetch_factor))
        return iter(())

    monkeypatch.setattr("flower.data.fineweb_token_stream", fake_fineweb_token_stream)
    device = torch.device("cpu")

    assert token_batches(DataConfig(dataset="fineweb_edu"), 2, device) is not None
    assert token_batches(DataConfig(dataset="fineweb-edu"), 3, device) is not None

    assert [call[0].dataset for call in calls] == ["fineweb_edu", "fineweb-edu"]
    assert [call[1] for call in calls] == [2, 3]
    assert all(call[2] == device for call in calls)
    # The train path must forward the DataConfig worker settings. It previously
    # dropped them and silently ran on _fineweb_loader's 2-worker default.
    assert all(call[3] == DataConfig.num_workers for call in calls)
    assert all(call[4] == DataConfig.prefetch_factor for call in calls)


def test_train_path_forwards_configured_worker_count(monkeypatch):
    """data.num_workers reaches the training DataLoader (regression guard)."""
    seen: dict[str, int | None] = {}

    def fake_loader(config, batch_size, device, split_role, *, num_workers=None, prefetch_factor=None):
        seen["role"] = split_role
        seen["num_workers"] = num_workers
        seen["prefetch_factor"] = prefetch_factor
        return iter(())

    monkeypatch.setattr("flower.data._fineweb_loader", fake_loader)
    cfg = DataConfig(dataset="fineweb_edu", num_workers=8, prefetch_factor=6)
    token_batches(cfg, 2, torch.device("cpu"))

    assert seen == {"role": "train", "num_workers": 8, "prefetch_factor": 6}


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


def _cache_shards() -> list[str]:
    from glob import glob

    return sorted(glob("data_cache/sample/10BT/*.parquet"))


@pytest.mark.skipif(not _cache_shards(), reason="local FineWeb-Edu parquet cache not present")
def test_local_parquet_doc_iter_matches_datasets_streaming():
    """The direct-pyarrow cache path yields the identical document sequence to
    the datasets streaming path it replaced (2026-08-16 RAM fix: datasets
    streaming held ~2.5 GB resident per dataloader worker; pyarrow ~40 MB).

    Training comparability across the swap depends on this — token order is a
    function of document order. 500 docs spans several 1,000-row row-group
    batches; cross-file order is the sorted shard list both paths consume.
    """
    pytest.importorskip("datasets")
    from datasets import load_dataset

    from flower.data import _local_parquet_doc_iter

    files = _cache_shards()
    reference = load_dataset("parquet", data_files=files, split="train", streaming=True)
    reference = (row["text"] for row in reference if row["text"])
    ours = _local_parquet_doc_iter(files, "text")

    compared = 0
    for ref_text, ours_text in zip(reference, ours, strict=False):
        assert ref_text == ours_text
        compared += 1
        if compared >= 500:
            break
    assert compared == 500, f"only {compared} comparable docs — a path ended early"


@pytest.mark.skipif(
    not _cache_shards() or not Path("tokenizers/fineweb_16k.json").exists(),
    reason="local parquet cache or production tokenizer not present",
)
def test_chunk_stream_cache_path_deterministic(monkeypatch):
    """Smoke: the cache-backed chunk stream is deterministic across fresh
    iterators (order comes purely from the sorted shard list — no seed)."""
    import itertools

    from flower.data import _FineWebChunkStream

    monkeypatch.setenv("FLOWER_DATA_CACHE", "data_cache")
    cfg = DataConfig(
        dataset="fineweb_edu",
        tokenizer="custom:tokenizers/fineweb_16k.json",
        sequence_length=512,
    )

    def first_chunks(n: int) -> list[torch.Tensor]:
        return [c.clone() for c in itertools.islice(iter(_FineWebChunkStream(cfg, "train")), n)]

    a, b = first_chunks(3), first_chunks(3)
    assert len(a) == 3
    assert all(c.shape == (512,) and c.dtype == torch.long for c in a)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
