#!/usr/bin/env python3
"""Compare tokenization *algorithms* at a fixed vocabulary size.

Sweeping vocab size only explores one axis. This trains several genuinely
different tokenizers at the same vocab and measures them on held-out text.

  uv run python scripts/compare_tokenizer_algorithms.py --vocab 16384 \
      --cache data_cache/sample/10BT --docs 40000 --out tokenizers/algo

Algorithms
----------
* **bpe_regex** — byte-level BPE with GPT-2 pre-tokenization. The current
  recipe, and the baseline here. The regex forbids merges across whitespace and
  punctuation boundaries, so every token sits inside one "word".

* **bpe_noregex** — the same BPE with the pre-tokenization split removed, so
  merges may span whitespace and a single token can cover several words
  ("superword" tokens). This is the t=0 limit of SuperBPE (arXiv:2503.13423),
  which observes that the whitespace constraint is an assumption rather than a
  requirement, and that lifting it buys a large encoding-efficiency gain.
  SuperBPE proper is two-stage — learn with the constraint up to t merges, then
  without — and reports that t=0 is *worse* than a well-chosen transition, so
  treat this as a lower bound on the idea rather than a fair test of it.

* **unigram** — the Unigram LM tokenizer (Kudo 2018). Bostrom & Durrett 2020
  ("Byte Pair Encoding is Suboptimal for Language Model Pretraining") found it
  produces more morphologically plausible segmentations than BPE and improves
  downstream task performance at equal vocab. It picks a segmentation by
  likelihood under a unigram model rather than by greedy merge order.

* **wordpiece** — likelihood-ratio merges (BERT). Included as a reference point.

Metrics are the same as scripts/analyze_tokenizer.py: bytes/token (compression),
fertility (tokens per word), and Renyi efficiency at alpha=2.5 (how evenly the
vocabulary is used — Zouhar et al. 2023 found this predicts downstream quality
where compression alone does not).

CAVEAT: none of these metrics is downstream loss. They rank tokenizers cheaply
so that at most one or two candidates need an actual training run. Compression
and Renyi efficiency routinely disagree; when they do, the tie is broken by
training, comparing on **bits-per-byte** (never perplexity, which is not
comparable across tokenizers).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analyze_tokenizer import load_docs, renyi_efficiency  # noqa: E402  (sibling script)


def build_trainer(algo: str, vocab: int):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    if algo in {"bpe_regex", "bpe_noregex"}:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=(algo == "bpe_regex")
        )
        trainer = trainers.BpeTrainer(vocab_size=vocab, initial_alphabet=alphabet, show_progress=False)
    elif algo == "unigram":
        tok = Tokenizer(models.Unigram())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab, initial_alphabet=alphabet, show_progress=False, unk_token=None
        )
    elif algo == "wordpiece":
        tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab, initial_alphabet=alphabet, show_progress=False, special_tokens=["[UNK]"]
        )
    else:
        raise ValueError(f"unknown algorithm {algo!r}")
    return tok, trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab", type=int, default=16384)
    parser.add_argument("--cache", type=Path, default=Path("data_cache/sample/10BT"))
    parser.add_argument("--docs", type=int, default=40000, help="Training documents")
    parser.add_argument("--eval-docs", type=int, default=2000, help="Held-out documents")
    parser.add_argument("--out", type=Path, default=Path("tokenizers/algo"))
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["bpe_regex", "bpe_noregex", "unigram", "wordpiece"],
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from collections import Counter

    all_docs = load_docs(args.cache, args.eval_docs + args.docs)
    eval_docs, train_docs = all_docs[: args.eval_docs], all_docs[args.eval_docs :]
    text = "".join(eval_docs)
    n_bytes, n_words = len(text.encode("utf-8")), len(text.split())
    print(f"train on {len(train_docs)} docs, measure on {len(eval_docs)} held-out ({n_bytes / 1e6:.1f} MB)\n")

    header = f"{'algorithm':<14}{'vocab':>7}{'bytes/tok':>11}{'fertility':>11}{'renyi':>8}{'unused':>8}"
    print(header)
    print("-" * len(header))
    for algo in args.algos:
        tok, trainer = build_trainer(algo, args.vocab)
        tok.train_from_iterator(iter(train_docs), trainer=trainer, length=len(train_docs))
        path = args.out / f"{algo}_{args.vocab}.json"
        tok.save(str(path))
        ids = tok.encode(text).ids
        counts = Counter(ids)
        v = tok.get_vocab_size()
        print(
            f"{algo:<14}{v:>7}{n_bytes / len(ids):>11.3f}{len(ids) / max(n_words, 1):>11.3f}"
            f"{renyi_efficiency(counts, v):>8.3f}{v - len(counts):>8}"
        )

    print(f"\nSaved to {args.out}/. Point a config at one with:")
    print(f'  data:\n    tokenizer: "custom:{args.out}/<algo>_{args.vocab}.json"')
    print("\nCompression and Renyi routinely disagree — break the tie with a training")
    print("run compared on val_bpb, not val_perplexity.")


if __name__ == "__main__":
    main()
