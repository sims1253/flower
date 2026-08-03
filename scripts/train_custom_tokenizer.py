#!/usr/bin/env python3
"""Train a small BPE tokenizer on a sample of FineWeb-Edu.

Usage:
  uv run python scripts/train_custom_tokenizer.py --vocab-size 4096 --num-docs 20000 \
      --output tokenizers/fineweb_4k.json

  # offline, from the pre-fetched parquet shards (see scripts/prefetch_dataset.py)
  uv run python scripts/train_custom_tokenizer.py --vocab-size 32768 \
      --from-cache data_cache/sample/10BT --output tokenizers/fineweb_32k.json

The output JSON file is loadable by `flower.data.build_tokenizer("custom:<path>")`.

Vocabulary size is a data-efficiency lever, not just a parameter-count knob: a
small vocab packs fewer bytes into the same sequence length, so a fixed token
budget sees proportionally less text and a fixed context window covers less
document. The script reports bytes/token on held-out docs so the trade against
the (tied) embedding cost can be made on measured numbers.
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

# Docs held out of training and used only for the compression report.
REPORT_DOCS = 2000


def _cache_doc_iter(cache_dir: Path):
    """Yield document texts from pre-fetched parquet shards."""
    import pyarrow.parquet as pq

    shards = sorted(cache_dir.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards under {cache_dir}")
    for shard in shards:
        table = pq.read_table(shard, columns=["text"])
        for text in table.column("text").to_pylist():
            if text:
                yield text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=4096, help="Target BPE vocabulary size")
    parser.add_argument("--num-docs", type=int, default=20000, help="Number of FineWeb-Edu documents to train on")
    parser.add_argument("--output", type=Path, default=Path("tokenizers/fineweb_4k.json"))
    parser.add_argument("--specials", action="store_true", help="Include <unk>, <bos>, <eos> as special tokens")
    parser.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        help="Read docs from local parquet shards instead of streaming from HF",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    from tokenizers import Tokenizer

    if args.from_cache is not None:
        print(f"Reading docs from {args.from_cache}...")
        source = _cache_doc_iter(args.from_cache)
    else:
        from datasets import load_dataset

        print("Streaming docs from FineWeb-Edu sample-10BT...")
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        source = (row["text"] for row in ds)

    # Hold out the first REPORT_DOCS for the compression report so bytes/token is
    # measured on text the merges were not fitted to.
    report_docs = list(islice(source, REPORT_DOCS))

    def text_iter():
        yield from islice(source, args.num_docs)

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    special_tokens = ["<unk>", "<bos>", "<eos>"] if args.specials else []
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),  # ensures bytes are coverable
        show_progress=True,
    )
    print(f"Training BPE with vocab_size={args.vocab_size}...")
    tokenizer.train_from_iterator(text_iter(), trainer=trainer, length=args.num_docs)

    tokenizer.save(str(args.output))
    actual_vocab = tokenizer.get_vocab_size()
    print(f"Saved to {args.output} (final vocab_size={actual_vocab})")

    sample = "Education is the most powerful weapon you can use to change the world."
    enc = tokenizer.encode(sample)
    print(f"Sample encoding ({len(enc.ids)} tokens for {len(sample)} chars):")
    print(f"  ids: {enc.ids[:20]}{'...' if len(enc.ids) > 20 else ''}")
    print(f"  tokens: {enc.tokens[:20]}{'...' if len(enc.tokens) > 20 else ''}")

    if report_docs:
        text = "".join(report_docs)
        n_tokens = len(tokenizer.encode(text).ids)
        n_bytes = len(text.encode("utf-8"))
        print(
            f"\nHeld-out compression over {len(report_docs)} docs: "
            f"{n_bytes / n_tokens:.3f} bytes/token "
            f"({n_bytes / 1e6:.1f} MB -> {n_tokens / 1e6:.2f} M tokens)"
        )


if __name__ == "__main__":
    main()
