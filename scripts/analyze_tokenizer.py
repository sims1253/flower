#!/usr/bin/env python3
"""Compare tokenizers on held-out text, and size the vocab against the model.

  uv run python scripts/analyze_tokenizer.py --cache data_cache/sample/10BT \
      tokenizers/fineweb_4k.json tokenizers/fineweb_16k.json tokenizers/fineweb_32k.json

Reports, per tokenizer:

* **bytes/token** — raw compression. More bytes per token means the same token
  budget covers more text at the same FLOPs, and the same context window covers
  more document. This is the metric that motivated the 4k -> 16k switch.
* **fertility** — tokens per whitespace word. The readable form of the same
  quantity; useful for spotting a tokenizer that compresses well on average but
  shreds ordinary words.
* **Renyi efficiency (alpha=2.5)** — how evenly the token distribution uses the
  vocabulary, following Zouhar et al. 2023 ("Tokenization and the Noiseless
  Channel"). Compression alone is a poor predictor of downstream quality: a
  tokenizer can win on bytes/token while dumping most of its probability mass
  onto a few high-frequency merges, leaving the rest of the vocabulary rarely
  trained. Renyi efficiency penalises exactly that, and correlated with
  downstream BLEU where compression did not. Higher is better; 1.0 is a uniform
  distribution over the vocabulary.
* **unused / rare** — merges that never appear, or appear <10 times, in the
  sample. A large tail here means embedding rows that will barely receive
  gradient. It is the concrete cost of over-sizing the vocabulary.

With `--d-model` it also prints the tied-embedding parameter cost and the
resulting text budget, which is the actual trade being made.

NOTE ON COMPARING RUNS: perplexity is NOT comparable across tokenizers — a
coarser tokenizer predicts more text per token, so its per-token loss is higher
for reasons unrelated to model quality. Use bits-per-byte. Paste the measured
bytes/token into `data.bytes_per_token` in the config and training-time
validation will emit `val_bpb` alongside `val_perplexity`.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

DEFAULT_ALPHA = 2.5  # Zouhar et al. report alpha=2.5 as the best-correlating setting.


def load_docs(cache_dir: Path, limit: int) -> list[str]:
    import pyarrow.parquet as pq

    shards = sorted(cache_dir.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards under {cache_dir}")
    docs: list[str] = []
    for shard in shards:
        for text in pq.read_table(shard, columns=["text"]).column("text").to_pylist():
            if text:
                docs.append(text)
            if len(docs) >= limit:
                return docs
    return docs


def renyi_efficiency(counts: Counter, vocab_size: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Renyi entropy of the token distribution, normalised by log(vocab_size).

    alpha > 1 weights the head of the distribution more heavily than Shannon
    entropy does, so a tokenizer that concentrates mass on a few merges scores
    lower even if its raw compression is good.
    """
    total = sum(counts.values())
    if total == 0 or vocab_size <= 1:
        return 0.0
    probs = [c / total for c in counts.values()]
    if abs(alpha - 1.0) < 1e-9:
        entropy = -sum(p * math.log(p) for p in probs)
    else:
        entropy = math.log(sum(p**alpha for p in probs)) / (1.0 - alpha)
    return entropy / math.log(vocab_size)


def analyze(path: Path, docs: list[str], alpha: float) -> dict:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    vocab = tok.get_vocab_size()
    text = "".join(docs)
    ids = tok.encode(text).ids
    n_bytes = len(text.encode("utf-8"))
    n_words = len(text.split())
    counts = Counter(ids)
    return {
        "name": path.stem,
        "vocab": vocab,
        "bytes_per_token": n_bytes / len(ids),
        "fertility": len(ids) / max(n_words, 1),
        "renyi": renyi_efficiency(counts, vocab, alpha),
        "unused": vocab - len(counts),
        "rare": sum(1 for v in counts.values() if v < 10),
        "tokens": len(ids),
        "bytes": n_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tokenizers", nargs="+", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("data_cache/sample/10BT"))
    parser.add_argument("--docs", type=int, default=2000, help="Held-out documents to measure on")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--d-model", type=int, default=None, help="Report tied-embedding cost at this width")
    parser.add_argument("--token-budget", type=int, default=None, help="Training tokens, for the text-budget column")
    args = parser.parse_args()

    docs = load_docs(args.cache, args.docs)
    rows = [analyze(p, docs, args.alpha) for p in args.tokenizers]

    header = f"{'tokenizer':<20}{'vocab':>7}{'bytes/tok':>11}{'fertility':>11}{'renyi':>8}{'unused':>8}{'rare<10':>9}"
    if args.d_model:
        header += f"{'embed':>9}"
    if args.token_budget and args.d_model:
        header += f"{'text seen':>11}"
    print(f"Measured on {len(docs)} held-out docs ({rows[0]['bytes'] / 1e6:.1f} MB)\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (
            f"{r['name']:<20}{r['vocab']:>7}{r['bytes_per_token']:>11.3f}{r['fertility']:>11.3f}"
            f"{r['renyi']:>8.3f}{r['unused']:>8}{r['rare']:>9}"
        )
        if args.d_model:
            line += f"{r['vocab'] * args.d_model / 1e6:>8.1f}M"
        if args.token_budget and args.d_model:
            line += f"{args.token_budget * r['bytes_per_token'] / 1e9:>10.2f}GB"
        print(line)

    if args.token_budget:
        # "rare in the sample" is not "rare in training" — scale it out, or the
        # rare-token column reads far more alarming than it is.
        scale = args.token_budget / max(rows[0]["tokens"], 1)
        print(
            f"\nSample is {scale:.0f}x smaller than a {args.token_budget / 1e6:.0f}M-token run, so a "
            f"token seen <10 times here is seen ~{10 * scale:,.0f} times in training. The "
            "`unused` column is the one to weigh; `rare<10` is mostly a small-sample artifact."
        )

    best = max(rows, key=lambda r: r["bytes_per_token"])
    print(f"\nPaste into the config of whichever you pick, e.g. for {best['name']}:")
    print(f"  data:\n    bytes_per_token: {best['bytes_per_token']:.3f}")
    print("\nReminder: compare runs across tokenizers on val_bpb, never val_perplexity.")


if __name__ == "__main__":
    main()
