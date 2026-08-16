#!/usr/bin/env python3
"""Sweep FlexAttention block sizes and kernel options at the 450M shape.

WHY
  The baseline profile (docs/profiling/baseline_profile.md) puts FlexAttention
  at 23.0% of the 450M vanilla step's kernel time, and the single largest kernel
  in the whole step is its backward:

      998 ms/step  triton_tem_fused_..._flex_attention_backward_split_transpose
      288 ms/step  triton_tem_fused_..._flex_attention_split_transpose

  FP8 does nothing for this — it is a Triton attention template, not a cutlass
  GEMM. After FP8 takes the GEMM bucket down, attention is the largest remaining
  block of step time, so it is where the next win has to come from.

WHAT IS BEING SWEPT
  `make_causal_local_block_mask` currently calls `create_block_mask` with the
  default 128x128 block granularity and passes no `kernel_options`, i.e. the
  defaults have never been measured against alternatives for THIS shape
  (seq 8192, window 2048, head_dim 64, 20 heads).

  Two knobs interact:
    - BLOCK_SIZE (mask granularity). Coarser blocks mean fewer mask entries and
      less bookkeeping, but a sliding-window edge is then approximated at a
      coarser step so more fully-masked work gets computed anyway. Finer blocks
      track the window tightly but multiply the block count.
    - kernel_options BLOCK_M/BLOCK_N/num_warps/num_stages (the Triton launch
      geometry).

  Correctness is checked against the current default configuration on every
  candidate: a faster kernel that computes a different attention is not a
  speedup, and a mismatched block size silently changes the mask.

USAGE
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
    uv run python scripts/bench_flex_config.py
  ... --seq 8192 --window 2048 --heads 20 --head-dim 64 --batch 2
"""

from __future__ import annotations

import argparse
import itertools
import time

import torch


def build(seq: int, window: int, block_size, device):
    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) < window)

    kwargs = {}
    if block_size is not None:
        kwargs["BLOCK_SIZE"] = block_size
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq, KV_LEN=seq, device=device, **kwargs)


def timed(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=20)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    from torch.nn.attention.flex_attention import flex_attention

    # Each (BLOCK_SIZE, kernel_options) pair is a distinct compilation. The
    # default recompile limit is 8; this sweep has ~45 combinations, and once
    # the limit is hit Dynamo silently falls back to the EAGER flex path, which
    # materialises the dense (B, H, T, T) scores — at this shape that is both
    # wrong as a measurement and an OOM risk. Raise it above the combination
    # count so every candidate is actually measured compiled.
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 512

    dev = torch.device("cuda")
    B, H, T, D = args.batch, args.heads, args.seq, args.head_dim
    torch.manual_seed(0)

    def mk():
        return [
            torch.randn(B, H, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        ]

    q, k, v = mk()
    grad_out = torch.randn_like(q)
    flex_c = torch.compile(flex_attention, dynamic=False)

    print(f"shape B={B} H={H} T={T} D={D} window={args.window} bf16, fwd+bwd\n")

    # Reference: current production settings (default block size, no kernel opts).
    ref_mask = build(T, args.window, None, dev)
    out_ref = flex_c(q, k, v, block_mask=ref_mask)
    ref_val = out_ref.detach().clone()

    def run(mask, kopts):
        def step():
            for t in (q, k, v):
                t.grad = None
            o = flex_c(q, k, v, block_mask=mask, kernel_options=kopts) if kopts else flex_c(q, k, v, block_mask=mask)
            o.backward(grad_out)
        return step

    baseline_ms = timed(run(ref_mask, None), args.warmup, args.iters)
    print(f"{'block_size':>12} {'BLOCK_M':>8} {'BLOCK_N':>8} {'warps':>6} {'stages':>7} {'ms':>8} {'vs base':>9}  ok")
    print(f"{'128 (default)':>12} {'-':>8} {'-':>8} {'-':>6} {'-':>7} {baseline_ms:8.2f} {1.0:8.3f}x  ref")

    block_sizes = [64, 128, 256]
    kernel_opts = [
        None,
        {"BLOCK_M": 64, "BLOCK_N": 64},
        {"BLOCK_M": 128, "BLOCK_N": 64},
        {"BLOCK_M": 64, "BLOCK_N": 128},
        {"BLOCK_M": 128, "BLOCK_N": 128},
    ]
    warp_stage = [None, {"num_warps": 8, "num_stages": 3}, {"num_warps": 4, "num_stages": 2}]

    best = (baseline_ms, "default")
    for bs, kopt, ws in itertools.product(block_sizes, kernel_opts, warp_stage):
        opts = None
        if kopt or ws:
            opts = {**(kopt or {}), **(ws or {})}
        try:
            mask = build(T, args.window, bs, dev)
            # Correctness before speed: a different mask granularity that changes
            # the attention pattern is a bug, not a tuning result.
            o = flex_c(q, k, v, block_mask=mask, kernel_options=opts) if opts else flex_c(q, k, v, block_mask=mask)
            ok = torch.allclose(o.detach(), ref_val, atol=2e-2, rtol=2e-2)
            ms = timed(run(mask, opts), args.warmup, args.iters)
        except Exception as e:
            print(f"{bs:>12} {str(kopt):>8.8} {'':>8} {'':>6} {'':>7} {'FAIL':>8}  {type(e).__name__}: {str(e)[:50]}")
            continue
        m = (opts or {}).get("BLOCK_M", "-")
        n = (opts or {}).get("BLOCK_N", "-")
        w = (opts or {}).get("num_warps", "-")
        s = (opts or {}).get("num_stages", "-")
        flag = "ok" if ok else "MISMATCH"
        print(f"{bs:12} {str(m):>8} {str(n):>8} {str(w):>6} {str(s):>7} {ms:8.2f} {baseline_ms / ms:8.3f}x  {flag}")
        if ok and ms < best[0]:
            best = (ms, f"BLOCK_SIZE={bs} kernel_options={opts}")

    print(f"\nbest correct config: {best[1]}  ({best[0]:.2f} ms, {baseline_ms / best[0]:.3f}x vs default)")
    print(
        "Attention is 23.0% of the 450M step's kernel time, so a Kx speedup here "
        f"is worth about {23.0 * (1 - 1 / max(baseline_ms / best[0], 1e-9)):.1f}% of total kernel time."
    )


if __name__ == "__main__":
    main()
