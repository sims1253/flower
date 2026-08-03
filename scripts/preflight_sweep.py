#!/usr/bin/env python3
"""Preflight every arm of a sweep config before spending GPU hours on it.

  uv run python scripts/preflight_sweep.py configs/sweep13_*.yaml

Builds each variant on CPU and checks the things that fail *late* and expensively
otherwise:

* the model actually constructs (bad schedule lengths, invalid enum values)
* the tokenizer file exists
* **model.vocab_size matches the tokenizer's real vocab size** — a mismatch
  either crashes mid-run on an out-of-range embedding index, or silently wastes
  embedding rows, depending on which way it is wrong
* `bytes_per_token` is set, so runs emit `val_bpb` and stay comparable
* `still_pretrained_base`, when named, exists on disk
* per-arm parameter counts, so "budget-preserving" claims are visibly true
* the seed budget, so the run count is known before launching

Exit code is non-zero if any arm fails, so this can gate a launch script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flower.config import ExperimentConfig, load_config
from flower.models import build_model
from flower.models.base import count_parameters
from flower.sweep import load_sweep


def tokenizer_vocab(spec: str) -> tuple[int | None, str | None]:
    """Return (vocab_size, error) for a `data.tokenizer` spec."""
    if not spec.startswith("custom:"):
        return None, None  # hf/byte tokenizers: nothing cheap to check
    path = Path(spec.split("custom:", 1)[1])
    if not path.exists():
        return None, f"tokenizer file missing: {path}"
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(path)).get_vocab_size(), None
    except Exception as exc:  # pragma: no cover - depends on file contents
        return None, f"tokenizer unreadable ({path}): {exc}"


def check_arm(name: str, cfg: ExperimentConfig) -> tuple[bool, list[str], str]:
    problems: list[str] = []
    notes: list[str] = []

    vocab, err = tokenizer_vocab(cfg.data.tokenizer)
    if err:
        problems.append(err)
    elif vocab is not None and vocab != cfg.model.vocab_size:
        problems.append(f"vocab mismatch: model.vocab_size={cfg.model.vocab_size} but tokenizer has {vocab}")

    if cfg.data.dataset not in {"synthetic", "mqar"} and not cfg.data.bytes_per_token:
        notes.append("no bytes_per_token -> no val_bpb")

    base = getattr(cfg.model, "still_pretrained_base", None)
    if base and not Path(base).exists():
        problems.append(f"still_pretrained_base missing: {base}")

    params = ""
    try:
        model = build_model(cfg.model)
        total = count_parameters(model)
        emb = model.token.weight.numel() if hasattr(model, "token") else 0
        params = f"{total / 1e6:7.2f}M total / {(total - emb) / 1e6:6.2f}M core"
    except Exception as exc:
        problems.append(f"build failed: {type(exc).__name__}: {exc}")

    return not problems, problems + [f"note: {n}" for n in notes], params


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()

    failures = 0
    total_runs = 0
    for path in args.configs:
        raw = path.read_text()
        print(f"\n=== {path}")
        if "sweep:" in raw.split("\n#")[0] or raw.lstrip().startswith("sweep:") or "\nsweep:" in raw:
            _, variants = load_sweep(path)
            arms = [(v["name"], v["config"]) for v in variants]
        else:
            arms = [(path.stem, None)]

        for name, raw_cfg in arms:
            if raw_cfg is None:
                cfg = load_config(path)
                seeds = [cfg.training.seed]
            else:
                cfg = load_config(None, raw_cfg)
                seeds = list(raw_cfg.get("training", {}).get("seeds", [cfg.training.seed]))
            ok, msgs, params = check_arm(name, cfg)
            total_runs += len(seeds)
            status = "ok  " if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"  [{status}] {name:<24} {params}  x{len(seeds)} seed(s)")
            for m in msgs:
                print(f"           {m}")

    print(f"\n{total_runs} total runs across {len(args.configs)} config(s); {failures} arm(s) failing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
