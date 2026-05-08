from __future__ import annotations

from collections.abc import Iterator

import torch

from flower.config import DataConfig


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


def fineweb_token_stream(config: DataConfig, batch_size: int, device: torch.device, split: str | None = None) -> Iterator[torch.Tensor]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=split or config.split, streaming=config.streaming)
    buffer: list[int] = []
    batch: list[torch.Tensor] = []
    for row in dataset:
        buffer.extend(tokenizer.encode(row[config.text_field]))
        while len(buffer) >= config.sequence_length:
            chunk = torch.tensor(buffer[: config.sequence_length], dtype=torch.long)
            del buffer[: config.sequence_length]
            batch.append(chunk)
            if len(batch) == batch_size:
                yield torch.stack(batch).to(device)
                batch.clear()


def token_batches(config: DataConfig, batch_size: int, device: torch.device, *, split: str | None = None, seed: int = 1234) -> Iterator[torch.Tensor]:
    if config.dataset == "synthetic":
        return iter(SyntheticTokenStream(config, batch_size, device, seed=seed))
    if config.dataset in {"fineweb_edu", "fineweb-edu"}:
        if split is None:
            return fineweb_token_stream(config, batch_size, device)
        return fineweb_token_stream(config, batch_size, device, split=split)
    raise ValueError(f"Unknown dataset: {config.dataset}")

def validation_token_batches(config: DataConfig, batch_size: int, device: torch.device) -> Iterator[torch.Tensor]:
    if config.dataset == "synthetic":
        return token_batches(config, batch_size, device, seed=config.validation_seed)
    if config.validation_split is None:
        raise ValueError("data.validation_split must be set for validation on non-synthetic datasets")
    if config.validation_split == config.split:
        raise ValueError("data.validation_split must differ from data.split to avoid evaluating on training data")
    return token_batches(config, batch_size, device, split=config.validation_split)
