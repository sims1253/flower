#!/usr/bin/env python3
"""Compare vocab sizes on matched training data — the compute-optimal-vocab probe.

  uv run python scripts/compare_vocab_sizes.py \
      --cache data_cache/sample/10BT --train-docs 20000 --report-docs 2000 \
      --vocabs 4096 8192 16384 24576 32768 \
      --out tokenizers/vocab_sweep --reference tokenizers/fineweb_16k.json

Trains byte-level BPE at each vocab size on the SAME corpus and evaluates all
of them — plus the production reference tokenizer — on the SAME held-out docs.
This mirrors scripts/train_custom_tokenizer.py's recipe exactly (ByteLevel
pre-tokenization, no special tokens, byte alphabet seeded): a vocab comparison
is only meaningful when everything but the vocab is held fixed. Running that
script once per size would not give that — its held-out set is "the first N
docs of the source", which varies with source order, so doc order would be
confounded with vocab size.

DECISION RULE — why bytes/token is only half the economics:
  A larger vocab packs more bytes per token (fewer transformer passes per byte
  of training text) but the tied embedding/head matmul grows with vocab and is
  deliberately NOT converted to FP8 (bf16, ~5-8% of the fp8_stack step). The
  net wall-clock-per-byte estimate printed for each vocab is

      net(v) = (bpt(v) / bpt(baseline)) / (1 + h * (v / baseline - 1))

  for head-time shares h in {5%, 8%}. Under ~3-4% net the axis is dead (inside
  run-to-run drift); ~8%+ justifies queueing the 600-step GPU quality screen
  (tokenizer swap + bytes_per_token via a sweep arm's `data:` block; BPB stays
  the comparable metric — perplexity never is, across tokenizers).

Metrics beyond bytes/token (fertility, Renyi efficiency at alpha=2.5) are the
same as scripts/analyze_tokenizer.py; compression and Renyi routinely disagree
and Renyi is the one with evidence of predicting downstream quality
(Zouhar et al. 2023). None of this replaces a training run — it decides
whether one is worth queueing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_tokenizer import load_docs, renyi_efficiency  # noqa: E402  (sibling script)


def train_bpe(train_docs: list[str], vocab_size: int):
    """Byte-level BPE, the exact train_custom_tokenizer.py recipe."""
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    from tokenizers import Tokenizer

    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(train_docs, trainer=trainer, length=len(train_docs))
    return tok


def evaluate(tok, report_docs: list[str]) -> dict[str, float]:
    text = "".join(report_docs)
    n_bytes = len(text.encode("utf-8"))
    enc = tok.encode(text)
    counts = Counter(enc.ids)
    n_words = sum(len(doc.split()) for doc in report_docs)
    return {
        "bytes_per_token": n_bytes / len(enc.ids),
        "tokens": len(enc.ids),
        "fertility_tokens_per_word": len(enc.ids) / max(n_words, 1),
        "renyi_efficiency_2p5": renyi_efficiency(counts, tok.get_vocab_size()),
    }


def net_gain(bpt: float, bpt_base: float, vocab: int, vocab_base: int, head_share: float) -> float:
    """Wall-clock-per-byte estimate vs the baseline vocab.

    Numerator: bytes covered per token processed (throughput of bytes at fixed
    token throughput). Denominator: step-time growth from the head GEMM, whose
    FLOPs scale linearly with vocab while everything else is fixed.
    """
    step_mult = 1.0 + head_share * (vocab / vocab_base - 1.0)
    return (bpt / bpt_base) / step_mult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data_cache/sample/10BT"))
    parser.add_argument("--train-docs", type=int, default=20000, help="Docs to fit merges on (matches the fineweb_16k recipe)")
    parser.add_argument("--report-docs", type=int, default=2000, help="Held-out docs for the compression report")
    parser.add_argument("--vocabs", type=int, nargs="+", default=[4096, 8192, 16384, 24576, 32768])
    parser.add_argument("--out", type=Path, default=Path("tokenizers/vocab_sweep"))
    parser.add_argument("--reference", type=Path, default=Path("tokenizers/fineweb_16k.json"), help="Production tokenizer, evaluated alongside for anchoring")
    parser.add_argument("--d-model", type=int, default=1280, help="For the tied-embedding param column")
    parser.add_argument("--baseline-vocab", type=int, default=16384)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    total = args.report_docs + args.train_docs
    print(f"Loading {total} docs from {args.cache} (first {args.report_docs} held out)...")
    docs = load_docs(args.cache, total)
    if len(docs) < total:
        raise RuntimeError(f"cache yielded only {len(docs)} docs, need {total}")
    report_docs, train_docs = docs[: args.report_docs], docs[args.report_docs :]

    from tokenizers import Tokenizer

    rows: list[dict] = []
    for vocab in args.vocabs:
        print(f"Training BPE vocab={vocab} on {len(train_docs)} docs...", flush=True)
        tok = train_bpe(train_docs, vocab)
        path = args.out / f"fineweb_{vocab}.json"
        tok.save(str(path))
        row = evaluate(tok, report_docs)
        row.update({"vocab": vocab, "source": "sweep", "path": str(path)})
        rows.append(row)
        print(f"  {vocab}: {row['bytes_per_token']:.3f} bytes/token", flush=True)

    if args.reference.exists():
        tok = Tokenizer.from_file(str(args.reference))
        row = evaluate(tok, report_docs)
        row.update({"vocab": tok.get_vocab_size(), "source": "reference", "path": str(args.reference)})
        rows.append(row)
        print(f"  reference {row['vocab']}: {row['bytes_per_token']:.3f} bytes/token", flush=True)

    base = next((r for r in rows if r["vocab"] == args.baseline_vocab and r["source"] == "sweep"), None)
    if base is None:
        raise RuntimeError(f"vocab {args.baseline_vocab} not in --vocabs; needed as the baseline")

    base_params_m = base["vocab"] * args.d_model / 1e6
    header = (
        f"{'vocab':>7} {'b/t':>6} {'Δb/t':>6} {'tok/word':>8} {'renyi':>6} "
        f"{'net@5%':>7} {'net@8%':>7} {'tiedM':>7} {'ΔM':>6}  source"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        delta = r["bytes_per_token"] / base["bytes_per_token"] - 1.0
        net5 = net_gain(r["bytes_per_token"], base["bytes_per_token"], r["vocab"], base["vocab"], 0.05) - 1.0
        net8 = net_gain(r["bytes_per_token"], base["bytes_per_token"], r["vocab"], base["vocab"], 0.08) - 1.0
        params_m = r["vocab"] * args.d_model / 1e6
        print(
            f"{r['vocab']:>7} {r['bytes_per_token']:>6.3f} {delta:>+6.1%} "
            f"{r['fertility_tokens_per_word']:>8.3f} {r['renyi_efficiency_2p5']:>6.3f} "
            f"{net5:>+7.1%} {net8:>+7.1%} {params_m:>7.1f} {params_m - base_params_m:>+6.1f}  {r['source']}"
        )

    payload = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "baseline_vocab": args.baseline_vocab,
        "rows": rows,
    }
    results_path = args.out / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nResults written to {results_path}")
    print("Decision rule: net < ~3-4% = dead axis; ~8%+ = queue the GPU screen.")


if __name__ == "__main__":
    main()
