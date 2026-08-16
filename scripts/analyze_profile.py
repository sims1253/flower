#!/usr/bin/env python3
"""Summarise docs/profiling/traces/*.trace.json into the tables for the writeup.

Reads the Chrome traces that scripts/profile_step.py exported and prints the
per-category CUDA-kernel-time breakdown (the ground truth — torch.profiler's
key_averages double-counts under cudagraph, but the raw trace kernel events do
not) plus the launch counts from summary.json. Run after profile_step.py.

  uv run python scripts/analyze_profile.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

TRACE_DIR = Path("docs/profiling/traces")


def categorise(name: str) -> str:
    n = name.lower()
    if "flex_attention" in n:
        return "attention: flex (local self-attn)"
    if "flash" in n:
        return "attention: flash (bloom cross-attn)"
    if "cutlass" in n and "gemm" in n:
        return "matmul: cutlass GEMM (FFN/proj/opt NS)"
    if "newton" in n or "schulz" in n:
        return "optimizer: Newton-Schulz"
    if "foreach" in n:
        return "optimizer: AdamW/foreach"
    # Fused triton kernels: the compiled RMSNorm/LayerNorm lowers into a triton
    # reduction kernel whose name contains 'mean','pow','rsqrt','layer_norm'.
    if "layer_norm" in n or "rmsnorm" in n or "rms" in n:
        return "norm"
    if ("mean" in n and "rsqrt" in n) or ("pow" in n and "rsqrt" in n):
        return "norm"
    if "silu" in n or "gelu" in n:
        return "activation (swiglu)"
    if "softmax" in n or "nll" in n:
        return "softmax / cross-entropy"
    if "embedding" in n or "gather" in n:
        return "embedding/gather"
    if any(k in n for k in ("mul", "add", "copy", "cast", "view", "slice", "cat", "clone", "to_copy")):
        return "elementwise/copy/fused-misc"
    return "other"


def kernel_breakdown(trace_path: Path, n_steps: int = 10) -> tuple[dict[str, float], float, int]:
    with trace_path.open() as f:
        tr = json.load(f)
    kernels = [e for e in tr["traceEvents"] if e.get("cat") == "kernel"]
    by_cat: dict[str, float] = defaultdict(float)
    for e in kernels:
        by_cat[categorise(e["name"])] += e.get("dur", 0) / 1e3 / n_steps  # us->ms/step
    total = sum(by_cat.values())
    return by_cat, total, len(kernels)


def main() -> None:
    summary_path = TRACE_DIR / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    by_variant = {r["variant"]: r for r in summary}

    print("=" * 92)
    print("CATEGORY BREAKDOWN (CUDA kernel time from trace events, ms per profiled accum=16 step)")
    print("=" * 92)
    variants = ["vanilla_matched", "bloom_memory"]
    cats: dict[str, dict[str, float]] = {}
    totals: dict[str, float] = {}
    for v in variants:
        tp = TRACE_DIR / f"{v}.trace.json"
        if not tp.exists():
            continue
        cats[v], totals[v], _ = kernel_breakdown(tp)
    all_cats = sorted(set().union(*[set(c) for c in cats.values()]),
                      key=lambda c: -max(cats[v].get(c, 0) for v in cats))
    hdr = f"{'category':40s}" + "".join(f" {v:>22s}" for v in variants)
    print(hdr)
    for c in all_cats:
        row = f"{c:40s}"
        for v in variants:
            ms = cats[v].get(c, 0.0)
            pct = 100 * ms / totals[v] if totals[v] else 0
            row += f" {ms:8.1f}ms ({pct:4.1f}%)"
        print(row)
    print(f"{'TOTAL kernel time':40s}" + "".join(f" {totals[v]:8.1f}ms{'':>11s}" for v in variants))
    print(f"{'wall-clock step (ms)':40s}" + "".join(f" {by_variant[v]['ms_per_step']:8.0f}ms{'':>11s}" for v in variants))

    print("\n" + "=" * 92)
    print("TOP-15 CUDA KERNELS BY TIME (per step, from trace)")
    print("=" * 92)
    for v in variants:
        tp = TRACE_DIR / f"{v}.trace.json"
        if not tp.exists():
            continue
        with tp.open() as f:
            tr = json.load(f)
        by_name: dict[str, list] = defaultdict(lambda: [0.0, 0])
        for e in tr["traceEvents"]:
            if e.get("cat") != "kernel":
                continue
            by_name[e["name"]][0] += e.get("dur", 0) / 1e3 / 10
            by_name[e["name"]][1] += 1
        print(f"\n--- {v} ---")
        print(f"{'ms/step':>9s} {'calls':>7s}  kernel")
        for name, (ms, n) in sorted(by_name.items(), key=lambda x: -x[1][0])[:15]:
            print(f"{ms:9.1f} {n:7d}  {name[:62]}")

    print("\n" + "=" * 92)
    print("LAUNCH COUNTS & THROUGHPUT (from summary.json)")
    print("=" * 92)
    print(f"{'variant':18s} {'ms/step':>8s} {'tok/s':>8s} {'peakGB':>7s} {'k/step':>7s} {'fwd':>5s} {'bwd':>5s} {'opt':>5s}")
    for v in variants:
        r = by_variant.get(v)
        if not r:
            continue
        kc = r["kernel_counts"]
        print(f"{v:18s} {r['ms_per_step']:8.0f} {r['tok_s']:8.0f} {r['peak_gb']:7.2f} "
              f"{kc['total']:7d} {kc['forward']:5d} {kc['backward']:5d} {kc['optimizer']:5d}")


if __name__ == "__main__":
    main()
