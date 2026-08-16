#!/usr/bin/env python3
"""Is an FP8 attention kernel worth writing for this shape? (feasibility only)

CONTEXT
  After FP8 takes the linear layers, attention is the largest remaining block of
  the 450M step: `flex_attention_backward` alone is 20.8% of CUDA time and the
  forward another 6.5%. FlexAttention does not support FP8, and block-size /
  kernel-option tuning was already measured as a dead end
  (scripts/bench_flex_config.py: best alternative 1.028x, inside noise). So the
  only remaining way to speed attention up is a custom FP8 attention kernel.

  That is a large, numerically delicate piece of work (a correct sliding-window
  flash-attention forward AND backward in FP8). This script exists to decide
  whether it could possibly pay off BEFORE any of it is written.

THE QUESTION
  FP8's measured advantage on this GPU came from LARGE GEMMs — 1.8x on
  (32768 x 1280) x (1280 x 6784)-shaped work. Attention's inner GEMMs are tiny by
  comparison: with head_dim 64 and a 128-row query tile, QK^T is
  (128 x 64) @ (64 x 128) and PV is (128 x 128) @ (128 x 64), batched over
  batch x heads x tiles. Small tiles are bandwidth- and launch-bound rather than
  tensor-core bound, and FP8's advantage largely evaporates there.

  So: measure batched bf16 vs FP8 matmul AT ATTENTION'S ACTUAL INNER SHAPES.

INTERPRETING THE RESULT
  Attention is ~27% of post-FP8 CUDA time. An FP8 attention kernel could at best
  capture the speedup measured here on the GEMM portion of that 27% — and real
  flash-attention kernels spend a large fraction of their time on softmax,
  online rescaling and memory movement that FP8 does not accelerate at all.

  Rule of thumb applied by this script: if the inner GEMMs do not show at least
  ~1.4x, the whole-kernel win cannot plausibly clear ~5% of step time, which is
  not worth a hand-written FP8 flash-attention forward+backward.

USAGE
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python \
    scripts/bench_fp8_attention_feasibility.py
"""

from __future__ import annotations

import argparse
import time

import torch


def timed(fn, warmup: int = 10, iters: int = 50) -> float:
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
    ap.add_argument(
        "--scale-divisor",
        type=int,
        default=1,
        help="divide the tile count by this to shrink VRAM use. The bf16-vs-FP8 RATIO at a "
        "fixed tile shape is what this script measures, and that ratio is stable once the "
        "GPU is saturated — so a divisor lets this run safely alongside a long training job. "
        "Use 1 when the GPU is free.",
    )
    args = ap.parse_args()

    dev = torch.device("cuda")
    # 450M vanilla_matched: batch 2, 20 heads, seq 8192, head_dim 64, window 2048.
    B, H, T, D, W = 2, 20, 8192, 64, 2048
    TILE = 128
    q_tiles = T // TILE
    kv_tiles_per_q = W // TILE  # sliding window: each query tile sees ~16 kv tiles
    full_batched = B * H * q_tiles * kv_tiles_per_q
    batched = max(256, full_batched // max(1, args.scale_divisor))

    print(f"450M attention inner GEMMs: B={B} H={H} T={T} D={D} window={W}, tile={TILE}")
    print(f"batched matmul count per attention call: {full_batched:,}")
    if batched != full_batched:
        print(f"running at 1/{args.scale_divisor} scale ({batched:,} tiles) to share the GPU")
    print()

    results = {}
    for name, (m, k, n) in {
        "QK^T  (tile x D) @ (D x tile)": (TILE, D, TILE),
        "PV    (tile x tile) @ (tile x D)": (TILE, TILE, D),
    }.items():
        a = torch.randn(batched, m, k, device=dev, dtype=torch.bfloat16)
        b = torch.randn(batched, k, n, device=dev, dtype=torch.bfloat16)
        bf16_ms = timed(lambda a=a, b=b: torch.bmm(a, b))

        # FP8 has no batched _scaled_mm; the closest apples-to-apples proxy is the
        # same total FLOPs as one large 2-D scaled_mm, which FLATTERS FP8 (it
        # removes all the small-tile overhead a real kernel would pay). If FP8
        # does not win even here, it certainly will not win inside the kernel.
        M = batched * m
        af = torch.randn(M, k, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        bfp = torch.randn(n, k, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        sa = torch.ones(M, 1, device=dev)
        sb = torch.ones(1, n, device=dev)
        try:
            fp8_ms = timed(
                lambda af=af, bfp=bfp, sa=sa, sb=sb: torch._scaled_mm(af, bfp.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            )
        except Exception as e:
            print(f"{name}: fp8 FAILED {type(e).__name__}: {str(e)[:90]}")
            continue

        # Same-FLOP bf16 reference for the flattened form, so the ratio is honest.
        a2 = torch.randn(M, k, device=dev, dtype=torch.bfloat16)
        b2 = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
        bf16_flat_ms = timed(lambda a2=a2, b2=b2: a2 @ b2.t())

        ratio = bf16_flat_ms / fp8_ms
        results[name] = ratio
        print(f"{name}")
        print(f"   batched bf16 bmm (real shape)   {bf16_ms:8.3f} ms")
        print(f"   flattened bf16 mm  (FP8-flattering) {bf16_flat_ms:8.3f} ms")
        print(f"   flattened fp8 scaled_mm             {fp8_ms:8.3f} ms   -> {ratio:.2f}x\n")

    if results:
        best = max(results.values())
        print(f"best FP8 advantage at attention's inner shapes: {best:.2f}x")
        if best < 1.4:
            print(
                "VERDICT: not worth writing an FP8 attention kernel. Even under an\n"
                "  upper-bound measurement that removes small-tile overhead entirely,\n"
                "  the GEMM speedup is too small to clear ~5% of step time once\n"
                "  softmax / online rescaling / memory movement (which FP8 does not\n"
                "  accelerate) are included."
            )
        else:
            print(
                "VERDICT: potentially worth prototyping. Note this is still an UPPER\n"
                "  BOUND — a real kernel pays small-tile overhead and gains nothing on\n"
                "  the softmax and rescaling portions."
            )


if __name__ == "__main__":
    main()
