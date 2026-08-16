#!/usr/bin/env python3
"""Compare replacement candidates for the production 16K tokenizer.

  uv run python scripts/compare_tokenizer_candidates.py \
      --cache data_cache/sample/10BT --corpus-sizes 100000 500000 \
      --eval-docs 2000 --out tokenizers/vocab_sweep

The production tokenizer (`tokenizers/fineweb_16k.json`, recipe from
2026-05, regenerated 2026-08-04) is byte-level BPE with merges fitted on only
20,000 FineWeb-Edu docs. Three families of candidate could beat it, all
evaluated here on the SAME held-out docs (first `--eval-docs` of the cache,
the split every tokenizer script in this repo uses):

1. MORE TRAINING DATA, same recipe/vocab — "is 20k docs undertrained?"
   Trained here at each size in `--corpus-sizes`; the existing
   `tokenizers/algo/bpe_regex_16384.json` artifact (trained 2026-08-03 with
   compare_tokenizer_algorithms.py defaults = 40k docs) adds a free point.
2. OTHER ALGORITHMS, same vocab — the remaining `tokenizers/algo/*` artifacts
   (bpe_noregex = the t=0 SuperBPE lower bound, unigram, wordpiece).
3. OFF-THE-SHELF tokenizers — `gpt2` (50,257 vocab) as the reference for what
   a heavily-engineered production tokenizer compresses to.

DECISION RULE (from docs/profiling/vocab_size_results.md):
  At the SAME vocab the head doesn't grow, so the net wall-clock-per-byte
  change IS the bytes/token change — no head-share penalty. A candidate needs
  roughly +2-3% bytes/token at vocab 16384 to be worth a GPU screen.
  A different-vocab candidate (gpt2) goes through the vocab economics:
  net(v) = (bpt(v)/bpt(16K)) / (1 + h*(v/16384 - 1)), head shares h of 5-8%;
  the vocab sweep already showed the knee at 16K, so these are expected to
  lose and are printed mainly to close the question with measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_tokenizer import load_docs  # noqa: E402  (sibling script)
from compare_vocab_sizes import evaluate, net_gain, train_bpe  # noqa: E402  (sibling script)

# Artifacts in tokenizers/algo/ were trained 2026-08-03 with
# compare_tokenizer_algorithms.py at its default --docs 40000.
_ALGO_TRAIN_DOCS = 40_000
_BASELINE_VOCAB = 16384


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data_cache/sample/10BT"))
    parser.add_argument(
        "--corpus-sizes",
        type=int,
        nargs="+",
        default=[100_000, 500_000],
        help="Docs to fit the production recipe's merges on, per size",
    )
    parser.add_argument("--eval-docs", type=int, default=2000)
    parser.add_argument("--vocab", type=int, default=16384)
    parser.add_argument("--out", type=Path, default=Path("tokenizers/vocab_sweep"))
    parser.add_argument("--reference", type=Path, default=Path("tokenizers/fineweb_16k.json"))
    parser.add_argument("--algo-dir", type=Path, default=Path("tokenizers/algo"))
    parser.add_argument("--pretrained", nargs="+", default=["gpt2"])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer

    max_docs = args.eval_docs + max(args.corpus_sizes)
    print(f"Loading {max_docs} docs from {args.cache} (first {args.eval_docs} held out)...")
    docs = load_docs(args.cache, max_docs)
    if len(docs) < max_docs:
        raise RuntimeError(f"cache yielded only {len(docs)} docs, need {max_docs}")
    held = docs[: args.eval_docs]

    rows: list[dict] = []

    def add(name: str, tok, train_docs: int | None, source: str) -> None:
        row = evaluate(tok, held)
        row.update({"name": name, "vocab": tok.get_vocab_size(), "train_docs": train_docs, "source": source})
        rows.append(row)
        print(f"  {name:<28} {row['bytes_per_token']:.3f} b/t (vocab {row['vocab']})", flush=True)

    # Baseline: the production tokenizer.
    add("bpe_regex (production 20k)", Tokenizer.from_file(str(args.reference)), 20_000, str(args.reference))

    # Family 2 + the free corpus-size point: the algo artifacts.
    algo_artifacts = {
        "bpe_regex_16384": ("bpe_regex (algo 40k)", _ALGO_TRAIN_DOCS),
        "bpe_noregex_16384": ("bpe_noregex (algo 40k)", _ALGO_TRAIN_DOCS),
        "unigram_16384": ("unigram (algo 40k)", _ALGO_TRAIN_DOCS),
        "wordpiece_16384": ("wordpiece (algo 40k)", _ALGO_TRAIN_DOCS),
    }
    for stem, (name, train_docs) in algo_artifacts.items():
        path = args.algo_dir / f"{stem}.json"
        if path.exists():
            add(name, Tokenizer.from_file(str(path)), train_docs, str(path))
        else:
            print(f"  [skip] {path} missing")

    # Family 1: same recipe, more data.
    for size in sorted(args.corpus_sizes):
        print(f"Training bpe_regex vocab={args.vocab} on {size:,} docs...", flush=True)
        tok = train_bpe(docs[args.eval_docs : args.eval_docs + size], args.vocab)
        path = args.out / f"fineweb_{args.vocab}_corpus{size}.json"
        tok.save(str(path))
        add(f"bpe_regex (corpus {size:,})", tok, size, str(path))

    # Family 3: off-the-shelf.
    for pretrained in args.pretrained:
        try:
            tok = Tokenizer.from_pretrained(pretrained)
        except Exception as e:  # network/hub unavailable — not worth failing the run
            print(f"  [skip] {pretrained}: {type(e).__name__}: {e}")
            continue
        add(f"{pretrained} (pretrained)", tok, None, pretrained)

    base = rows[0]
    base_bpt = base["bytes_per_token"]
    header = (
        f"{'candidate':<28}{'vocab':>7}{'train docs':>11}{'b/t':>7}{'Δb/t':>7} "
        f"{'net@5%':>7}{'net@8%':>7}{'tok/w':>7}{'renyi':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        delta = r["bytes_per_token"] / base_bpt - 1.0
        net5 = net_gain(r["bytes_per_token"], base_bpt, r["vocab"], base["vocab"], 0.05) - 1.0
        net8 = net_gain(r["bytes_per_token"], base_bpt, r["vocab"], base["vocab"], 0.08) - 1.0
        train_docs = f"{r['train_docs']:,}" if r["train_docs"] else "-"
        print(
            f"{r['name']:<28}{r['vocab']:>7}{train_docs:>11}{r['bytes_per_token']:>7.3f}{delta:>+7.1%} "
            f"{net5:>+7.1%}{net8:>+7.1%}{r['fertility_tokens_per_word']:>7.3f}{r['renyi_efficiency_2p5']:>7.3f}"
        )

    payload = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "rows": rows}
    results_path = args.out / "candidates_results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nResults written to {results_path}")
    print("Rule: same-vocab candidates need ~+2-3% b/t (net == Δb/t at fixed vocab);")
    print("different-vocab candidates go through the vocab_size_results.md economics.")


if __name__ == "__main__":
    main()
