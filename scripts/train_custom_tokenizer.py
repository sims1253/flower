#!/usr/bin/env python3
"""Train a small BPE tokenizer on a sample of FineWeb-Edu.

Usage:
  uv run python scripts/train_custom_tokenizer.py --vocab-size 4096 --num-docs 20000 \
      --output tokenizers/fineweb_4k.json

The output JSON file is loadable by `flower.data.build_tokenizer("custom:<path>")`.
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=4096, help="Target BPE vocabulary size")
    parser.add_argument("--num-docs", type=int, default=20000, help="Number of FineWeb-Edu documents to train on")
    parser.add_argument("--output", type=Path, default=Path("tokenizers/fineweb_4k.json"))
    parser.add_argument("--specials", action="store_true", help="Include <unk>, <bos>, <eos> as special tokens")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    from tokenizers import Tokenizer

    print(f"Streaming {args.num_docs} docs from FineWeb-Edu sample-10BT...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    def text_iter():
        for row in islice(ds, args.num_docs):
            yield row["text"]

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


if __name__ == "__main__":
    main()
