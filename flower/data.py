from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from flower.config import DataConfig


class _Encoder(Protocol):
    """Minimal interface for tokenizers used by the streaming chunker."""

    vocab_size: int

    def encode(self, text: str) -> list[int]: ...


class _ByteEncoder:
    """UTF-8 byte-level encoder. 256-symbol vocab, no special tokens."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))


class _CustomBPEEncoder:
    """Loads a custom BPE tokenizer trained via the `tokenizers` library."""

    def __init__(self, path: str) -> None:
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids


class _HFEncoderAdapter:
    """Wraps a transformers AutoTokenizer to expose the same minimal interface."""

    def __init__(self, name: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)


def build_tokenizer(name: str) -> _Encoder:
    """Resolve a tokenizer spec string into an encoder with `.encode()` and `.vocab_size`.

    Supported specs:
      - "byte"                    → byte-level (256 vocab, no training needed).
      - "custom:<path/to/json>"   → BPE loaded from a `tokenizers` save file.
      - anything else             → HuggingFace AutoTokenizer (e.g. "gpt2").
    """
    if name == "byte":
        return _ByteEncoder()
    if name.startswith("custom:"):
        path = name[len("custom:") :]
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Custom tokenizer file not found: {path}. Train one with scripts/train_custom_tokenizer.py."
            )
        return _CustomBPEEncoder(path)
    return _HFEncoderAdapter(name)


class SyntheticTokenStream:
    def __init__(self, config: DataConfig, batch_size: int, device: torch.device, seed: int = 1234) -> None:
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def __iter__(self) -> Iterator[torch.Tensor]:
        vocab = min(self.config.synthetic_vocab_size, 50257)
        seq = self.config.sequence_length
        while True:
            data = torch.randint(0, vocab, (self.batch_size, seq), generator=self.generator)
            yield data.to(self.device)


class MQARTokenStream:
    """Synthetic multi-query associative-recall stream.

    Each sequence is `[k1,v1,...,kN,vN] + delay + [q1,a1,q2,a2,...]` where
    `a_i` is the value paired with query key `q_i`. Labels are `-100` except at
    answer positions, so training is not dominated by unpredictable random
    keys/values/delay tokens.
    """

    def __init__(self, config: DataConfig, batch_size: int, device: torch.device, seed: int = 1234) -> None:
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def _sample_unique(self, vocab: int, count: int) -> torch.Tensor:
        if vocab >= count:
            return torch.randperm(vocab, generator=self.generator, dtype=torch.long)[:count]
        return torch.randint(0, vocab, (count,), generator=self.generator, dtype=torch.long)

    def __iter__(self) -> Iterator[torch.Tensor]:
        vocab = min(self.config.synthetic_vocab_size, 50257)
        seq_len = self.config.sequence_length
        # Keep the task feasible for short smoke configs while scaling with context.
        max_pairs = max(1, min(128, seq_len // 4, vocab // 2 if vocab > 1 else 1))
        pair_choices = [p for p in (8, 16, 32, 64, 128) if p <= max_pairs]
        if not pair_choices:
            pair_choices = [max_pairs]
        while True:
            rows: list[torch.Tensor] = []
            label_rows: list[torch.Tensor] = []
            for _ in range(self.batch_size):
                num_pairs = int(pair_choices[torch.randint(0, len(pair_choices), (), generator=self.generator)])
                num_queries = max(1, min(num_pairs, seq_len // 8))
                kv_len = num_pairs * 2
                qa_len = num_queries * 2
                delay_len = max(0, seq_len - kv_len - qa_len)
                keys = self._sample_unique(vocab, num_pairs)
                vals = self._sample_unique(vocab, num_pairs)
                kv = torch.stack([keys, vals], dim=-1).reshape(kv_len)
                query_indices = torch.randperm(num_pairs, generator=self.generator)[:num_queries]
                query_keys = keys[query_indices]
                query_vals = vals[query_indices]
                delay = (
                    torch.randint(0, vocab, (delay_len,), generator=self.generator, dtype=torch.long)
                    if delay_len > 0
                    else kv.new_empty(0)
                )
                qa = torch.stack([query_keys, query_vals], dim=-1).reshape(qa_len)
                row = torch.cat([kv, delay, qa])
                labels = torch.full((row.numel(),), -100, dtype=torch.long)
                ans_start = kv_len + delay_len + 1
                labels[ans_start : ans_start + qa_len : 2] = query_vals
                if row.numel() < seq_len:
                    pad = torch.randint(0, vocab, (seq_len - row.numel(),), generator=self.generator, dtype=torch.long)
                    row = torch.cat([row, pad])
                    labels = torch.cat([labels, torch.full((pad.numel(),), -100, dtype=torch.long)])
                rows.append(row[:seq_len])
                label_rows.append(labels[:seq_len])
            yield torch.stack(rows, dim=0).to(self.device), torch.stack(label_rows, dim=0).to(self.device)


VAL_DOCS = 1024
"""Number of FineWeb-Edu documents reserved for validation. Train iterators skip
the first VAL_DOCS rows; validation iterators take the first VAL_DOCS rows. This
gives a hermetic train/val split over the streaming dataset (since the upstream
`sample-10BT` config exposes only a `train` split)."""


def _local_parquet_doc_iter(files: list[str], text_field: str) -> Iterator[str]:
    """Stream documents from local parquet shards one row-group batch at a time.

    This deliberately bypasses `datasets.load_dataset("parquet", ...)` for the
    FLOWER_DATA_CACHE path: measured 2026-08-16, the datasets streaming stack
    holds ~2.5 GB resident PER DATALOADER WORKER at steady state (two workers
    per training run = ~5 GB of a 31 GB WSL VM, and each worker is a fork that
    pays it privately — the suspected contributor to a VM OOM that killed a
    12-hour run at 83%). Direct pyarrow iteration of the same shards plateaus
    ~2.2 GB lower per worker at identical document order (files in the given
    order, rows in parquet order), pinned bit-equal against the datasets path
    by tests/test_data.py::test_local_parquet_doc_iter_matches_datasets_streaming.
    Row groups in these shards are 1,000 rows (~5 MB), so `iter_batches` at
    1,000 streams one row group at a time.
    """
    import pyarrow.parquet as pq

    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1000, columns=[text_field]):
            for text in batch.column(text_field).to_pylist():
                if text:
                    yield text


class _FineWebChunkStream(IterableDataset):
    """Background-worker-friendly tokenized chunk stream over FineWeb-Edu.

    Each DataLoader worker holds its own HF streaming dataset iterator and tokenizer,
    skipping by `worker_id` modulo to avoid duplicate tokens across workers. Yields
    (sequence_length,) int64 token tensors; the DataLoader assembles the batch.

    `split_role`:
      - "train": skip the first VAL_DOCS docs, then iterate to infinity (looping if
        the streaming dataset eventually exhausts itself, which it won't at sample-10BT
        scale within a 30k-step run).
      - "validation": take only the first VAL_DOCS docs, looping forever to provide
        a fixed validation set.
    """

    def __init__(self, config: DataConfig, split_role: str) -> None:
        super().__init__()
        if split_role not in {"train", "validation"}:
            raise ValueError(f"split_role must be 'train' or 'validation', got {split_role!r}")
        self.config = config
        self.split_role = split_role

    def __iter__(self) -> Iterator[torch.Tensor]:
        import os
        from glob import glob
        from itertools import islice

        from datasets import load_dataset

        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        encoder = build_tokenizer(self.config.tokenizer)
        seq_len = self.config.sequence_length
        text_field = self.config.text_field

        # FLOWER_DATA_CACHE: pre-downloaded parquet shards (see scripts/prefetch_dataset.py).
        # When set, stream from local files instead of HF Hub HTTP — eliminates the
        # mid-run httpx-client-closed crashes that took down Phase 2 30k.
        cache_dir = os.environ.get("FLOWER_DATA_CACHE")
        local_parquets: list[str] = []
        if cache_dir:
            local_parquets = sorted(glob(f"{cache_dir}/sample/10BT/*.parquet"))
            if local_parquets and worker_id == 0:
                print(f"[data] using {len(local_parquets)} local parquet shards from {cache_dir}", flush=True)

        def doc_iter() -> Iterator[str]:
            # Outer loop: re-open the stream when it ends OR when streaming raises
            # transient HF/httpx errors. Phase 2 30k crashed mid-variant when the HF
            # httpx client closed mid-stream — recreating the dataset object lets the
            # run survive without losing the in-flight checkpoint.
            consecutive_failures = 0
            # Documents this worker has PULLED from its shard since this __iter__
            # began (empty texts included — they occupy a shard slot even though
            # they are never yielded). A restart re-opens the stream from the HEAD
            # of the region, so without this counter every already-yielded document
            # below would be yielded — and trained on — a second time, silently.
            # Skipping the same number of items from the rebuilt shard resumes this
            # worker's own document sequence exactly where it stopped (the re-opened
            # stream is assumed to yield the same order, which holds for the sorted
            # local-parquet path and for HF streaming order).
            #
            # Residual caveat, stated honestly: this preserves each worker's
            # document SEQUENCE, not the global batch interleaving. Batches mix
            # chunks from all workers in DataLoader arrival order, and after a
            # mid-run restart the workers' relative progress differs from an
            # uninterrupted run — the token SET is preserved, the interleaving is
            # not. That is the same class of difference config.py already documents
            # for changing num_workers: seed-comparable, not bit-comparable.
            docs_consumed = 0
            while True:
                try:
                    if local_parquets:
                        # Always streams (memory: see _local_parquet_doc_iter).
                        docs: Iterator[str] = _local_parquet_doc_iter(local_parquets, text_field)
                    else:
                        dataset = load_dataset(
                            "HuggingFaceFW/fineweb-edu",
                            name="sample-10BT",
                            split="train",
                            streaming=self.config.streaming,
                        )
                        docs = (row.get(text_field, "") for row in dataset)
                    if self.split_role == "validation":
                        sliced = islice(docs, VAL_DOCS)
                    else:  # train
                        sliced = islice(docs, VAL_DOCS, None)
                    sharded = islice(sliced, worker_id, None, num_workers)
                    if docs_consumed:
                        # Restart after an error: jump past the prefix of THIS
                        # worker's shard that was already consumed.
                        sharded = islice(sharded, docs_consumed, None)
                    for text in sharded:
                        # A document was successfully pulled, so the stream is
                        # healthy: count it and reset the failure streak HERE
                        # rather than after a complete pass (for the train role a
                        # pass never completes mid-run, so a pass-scoped reset
                        # made the counter cumulative — the 10th transient error
                        # of a whole run killed it, not the 10th back-to-back one).
                        docs_consumed += 1
                        consecutive_failures = 0
                        if text:
                            yield text
                    # Clean exhaustion (only reachable when the upstream dataset
                    # ends): the next pass is a fresh epoch over the region, so
                    # the skip counter must restart with it — otherwise every
                    # later pass would skip everything and silently yield
                    # nothing, forever.
                    docs_consumed = 0
                    consecutive_failures = 0
                except Exception as e:
                    consecutive_failures += 1
                    print(
                        f"[data] worker={worker_id} stream error (consecutive failures: "
                        f"{consecutive_failures}): {type(e).__name__}: {e}; restarting stream "
                        f"from the head and skipping the {docs_consumed} documents this "
                        f"worker already consumed",
                        flush=True,
                    )
                    if consecutive_failures >= 10:
                        # 10 back-to-back failures — something is structurally broken; let it die.
                        raise
                    import time

                    time.sleep(min(2**consecutive_failures, 60))

        buffer: list[int] = []
        for text in doc_iter():
            buffer.extend(encoder.encode(text))
            while len(buffer) >= seq_len:
                yield torch.tensor(buffer[:seq_len], dtype=torch.long)
                del buffer[:seq_len]


def fineweb_validation_documents(config: DataConfig, limit: int | None = None) -> Iterator[str]:
    """Yield the fixed FineWeb-Edu validation documents as raw UTF-8 text.

    This is intentionally document-oriented rather than chunk-oriented so eval can
    compute exact bits-per-byte with bootstrap CIs over documents.
    """
    import os
    from glob import glob
    from itertools import islice

    from datasets import load_dataset

    text_field = config.text_field
    take = VAL_DOCS if limit is None else min(limit, VAL_DOCS)

    cache_dir = os.environ.get("FLOWER_DATA_CACHE")
    local_parquets: list[str] = []
    if cache_dir:
        local_parquets = sorted(glob(f"{cache_dir}/sample/10BT/*.parquet"))

    if local_parquets:
        doc_source: Iterator[str] = _local_parquet_doc_iter(local_parquets, text_field)
    else:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=config.streaming,
        )
        doc_source = (row.get(text_field, "") for row in dataset)

    for text in islice(doc_source, take):
        if text:
            yield text


def _collate_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(chunks, dim=0)


def _fineweb_loader(
    config: DataConfig,
    batch_size: int,
    device: torch.device,
    split_role: str,
    *,
    num_workers: int = 2,
    prefetch_factor: int = 4,
) -> Iterator[torch.Tensor]:
    """DataLoader-backed FineWeb-Edu stream that overlaps tokenization with GPU work."""
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        _FineWebChunkStream(config, split_role),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=_collate_chunks,
        drop_last=True,
    )
    while True:
        for batch in loader:
            yield batch.to(device, non_blocking=pin_memory)


def fineweb_token_stream(
    config: DataConfig,
    batch_size: int,
    device: torch.device,
    *,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
) -> Iterator[torch.Tensor]:
    return _fineweb_loader(
        config,
        batch_size,
        device,
        "train",
        num_workers=config.num_workers if num_workers is None else num_workers,
        prefetch_factor=config.prefetch_factor if prefetch_factor is None else prefetch_factor,
    )


def token_batches(
    config: DataConfig,
    batch_size: int,
    device: torch.device,
    *,
    split: str | None = None,
    seed: int = 1234,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
) -> Iterator[torch.Tensor]:
    # None = take the value from DataConfig. Previously the train path ignored
    # these kwargs entirely and always ran on the _fineweb_loader default of 2
    # workers, so `data.num_workers` was silently inert for training.
    #
    # This was a plumbing bug, NOT a throughput bug: the loader was measured at
    # 1,079,869 tok/s with 2 workers against a ~52k tok/s consumption rate
    # (scripts/bench_data_workers.py), i.e. ~20x headroom. Raising the worker
    # count does not speed training up. See DataConfig.num_workers.
    workers = config.num_workers if num_workers is None else num_workers
    prefetch = config.prefetch_factor if prefetch_factor is None else prefetch_factor
    if config.dataset == "synthetic":
        return iter(SyntheticTokenStream(config, batch_size, device, seed=seed))
    if config.dataset == "mqar":
        return iter(MQARTokenStream(config, batch_size, device, seed=seed))
    if config.dataset in {"fineweb_edu", "fineweb-edu"}:
        # `split` is interpreted as a role here ("train" or "validation") since
        # FineWeb-Edu's sample-10BT only exposes a single upstream split.
        role = split or "train"
        if role not in {"train", "validation"}:
            role = "train"
        if role == "train":
            return fineweb_token_stream(
                config,
                batch_size,
                device,
                num_workers=workers,
                prefetch_factor=prefetch,
            )
        return _fineweb_loader(
            config,
            batch_size,
            device,
            role,
            num_workers=workers,
            prefetch_factor=prefetch,
        )
    raise ValueError(f"Unknown dataset: {config.dataset}")


def compress_to_bags(token_ids: torch.Tensor, bag_size: int) -> torch.Tensor:
    """Compress consecutive tokens into bags for TST phase 1 (arXiv:2605.06546).

    Input: (B, T) token IDs (or (T,) 1-D for a single sequence).
    Output: (B, T // bag_size, bag_size) — each output position holds the
    `bag_size` original tokens that will be averaged at the embedding level.

    The input is truncated to a multiple of `bag_size` so the reshape is exact.
    bag_size must be >= 1; bag_size == 1 is an identity (no compression).

    See `docs/training-speedups.md` Section 9 (Token Superposition Training).
    """
    if bag_size < 1:
        raise ValueError(f"bag_size must be >= 1, got {bag_size}")

    was_1d = token_ids.dim() == 1
    if was_1d:
        token_ids = token_ids.unsqueeze(0)  # (1, T)

    B, T = token_ids.shape
    t_compressed = T // bag_size
    token_ids = token_ids[..., : t_compressed * bag_size]

    if bag_size == 1:
        # Lossless identity: just add a trailing size-1 dim.
        out = token_ids.unsqueeze(-1)
    else:
        out = token_ids.view(B, t_compressed, bag_size)

    if was_1d:
        out = out.squeeze(0)  # (T_compressed, bag_size)
    return out


def validation_token_batches(
    config: DataConfig,
    batch_size: int,
    device: torch.device,
    *,
    num_workers: int = 1,
    prefetch_factor: int = 2,
) -> Iterator[torch.Tensor]:
    if config.dataset in {"synthetic", "mqar"}:
        return token_batches(config, batch_size, device, seed=config.validation_seed)
    if config.validation_split is not None and config.validation_split == config.split:
        raise ValueError("data.validation_split must differ from data.split")
    return token_batches(
        config,
        batch_size,
        device,
        split="validation",
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
