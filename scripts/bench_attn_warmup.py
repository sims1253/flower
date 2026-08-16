#!/usr/bin/env python3
"""Attention-window warmup: recompile safety first, throughput second.

WHY THIS IS NOT JUST ANOTHER bench_arms ARM
  `attn_warmup_steps` ramps `local_window` DURING training, so a bench that
  builds a model and times steps measures the final window and reports the
  control. The feature only exists across steps. This drives the real
  `update_attention_windows` on a compiled model exactly as train.py does.

THE THING THAT ACTUALLY DECIDES THIS FEATURE IS THE RECOMPILE COUNT
  `local_window` is a plain Python int on each attention module, so Dynamo
  guards on it: every distinct window is a fresh recompile of the model graph.
  Past `torch._dynamo.config.cache_size_limit` (default 8) Dynamo stops
  compiling that frame and drops to eager — and eager flex is the dense path
  this project moved off precisely because it OOMs at long context. So a warmup
  that is "faster" can in fact be a silent fallback.

  `attn_warmup_quantize` is the guard: it is a STEP-STRIDE, so the ramp visits
  at most ceil(warmup_steps / quantize) distinct windows. That count is
  computable without running anything, and this script reports it (and refuses
  to proceed without --force when it exceeds the limit) BEFORE spending GPU
  time.

  Caveat on that count: `cache_size_limit` is per code object, while the
  reported `unique_graphs` counts compiled frames across the whole model. A
  measured 14 unique graphs for 5 window changes on a 1-layer model therefore
  does NOT establish 2.8 recompiles of a single frame — the model has several
  distinct frames. The honest per-frame number is the `+N graphs` this script
  prints per plateau at runtime. Treat the distinct-window count as the
  planning proxy and the per-plateau delta as the measurement.

WHAT WAS CHECKED AND IS NOT A PROBLEM
  `update_attention_windows` nulls `_cached_block_mask` without rebuilding it,
  and `_get_or_build_block_mask` raises under `is_compiling()` on a null cache.
  That looks like a crash waiting to happen; it is not. Verified empirically:
  the mask is rebuilt on every window change and `_cached_window` tracks
  `local_window` correctly across a full ramp. Do not "fix" it.

USAGE
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
    uv run python scripts/bench_attn_warmup.py \
    --config configs/perf_bench_450m.yaml --arm perf_control \
    --warmup-steps 2000 --start 512 --quantize 250
  ... --plan-only        # print the window schedule + recompile risk, no GPU
  ... --accum 4
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time

import torch
from profile_step import build_everything, load_variant_config, make_synthetic_batch

from flower.train import update_attention_windows


def _autocast(amp_dtype):
    return torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()


def window_schedule(start: int, final: int, warmup_steps: int, quantize: int) -> list[tuple[int, int]]:
    """Reproduce update_attention_windows' ramp as (first_step, window) plateaus.

    Kept as an independent reimplementation rather than calling the trainer in a
    loop: the point is to know the schedule BEFORE touching the GPU, and a
    divergence between this and the trainer would itself be a finding. The
    bench asserts they agree at every measured plateau.
    """
    plateaus: list[tuple[int, int]] = []
    q = max(1, quantize)
    for step in range(1, warmup_steps + 1):
        eff = (step // q) * q
        frac = eff / max(1, warmup_steps)
        w = int(round(start + frac * (final - start)))
        if not plateaus or plateaus[-1][1] != w:
            plateaus.append((step, w))
    if not plateaus or plateaus[-1][1] != final:
        plateaus.append((warmup_steps, final))
    return plateaus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/perf_bench_450m.yaml")
    ap.add_argument("--arm", default="perf_control")
    ap.add_argument("--start", type=int, default=512, help="attn_warmup_start")
    ap.add_argument("--warmup-steps", type=int, default=2000, help="attn_warmup_steps")
    ap.add_argument("--quantize", type=int, default=250, help="attn_warmup_quantize (step stride)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--iters", type=int, default=4, help="timed steps per plateau")
    ap.add_argument("--settle", type=int, default=2, help="untimed steps per plateau (absorbs the recompile)")
    ap.add_argument(
        "--total-steps", type=int, default=None,
        help="real run length, for the whole-run projection (the bench config's `steps` is a placeholder)",
    )
    ap.add_argument("--plan-only", action="store_true", help="print the schedule and risk, touch no GPU")
    ap.add_argument("--force", action="store_true", help="run even if the plan exceeds cache_size_limit")
    args = ap.parse_args()

    cfg = load_variant_config(args.config, args.arm, accum_override=args.accum)
    final = int(cfg.model.local_window)
    plateaus = window_schedule(args.start, final, args.warmup_steps, args.quantize)
    limit = torch._dynamo.config.cache_size_limit

    print(f"arm {args.arm}: window {args.start} -> {final} over {args.warmup_steps} steps, quantize {args.quantize}")
    print(f"distinct windows: {len(plateaus)}   dynamo cache_size_limit: {limit}")
    print("  " + ", ".join(f"s{s}:w{w}" for s, w in plateaus))
    unsafe = len(plateaus) > limit
    if unsafe:
        safe_q = math.ceil(args.warmup_steps / max(1, limit - 1))
        print(
            f"\n  RISK: {len(plateaus)} distinct windows exceeds cache_size_limit {limit}.\n"
            f"  Past the limit Dynamo stops compiling and flex drops to the eager dense path,\n"
            f"  which is slower AND the OOM-prone one. The limit is per code object while\n"
            f"  this count is per distinct window, so treat it as a planning proxy: the\n"
            f"  per-plateau '+N graphs' column below is the real per-frame measurement.\n"
            f"  Raise --quantize to >= {safe_q}, or raise torch._dynamo.config.cache_size_limit."
        )
    if args.plan_only:
        return
    if unsafe and not args.force:
        print("\n  refusing to run; pass --force to measure it anyway")
        return

    accum, batch = args.accum, int(cfg.training.batch_size)
    seq, vocab = int(cfg.data.sequence_length), int(cfg.model.vocab_size)
    device, amp_dtype, eager_model, run_model, optims = build_everything(cfg)
    run_model.train()

    class _Cfg:
        pass

    ramp_cfg = _Cfg()
    ramp_cfg.model = cfg.model
    cfg.model.attn_warmup_start = args.start
    cfg.model.attn_warmup_steps = args.warmup_steps
    cfg.model.attn_warmup_quantize = args.quantize

    def one_step():
        for o in optims:
            o.zero_grad(set_to_none=True)
        for _ in range(accum):
            ids, labels = make_synthetic_batch(batch, seq, vocab, device)
            with _autocast(amp_dtype):
                loss = run_model(ids, labels=labels)["loss"]
            loss.backward()
        for o in optims:
            o.step()

    def graphs() -> int:
        return torch._dynamo.utils.counters["stats"]["unique_graphs"]

    rows = []
    for first_step, want_window in plateaus:
        update_attention_windows(eager_model, first_step, ramp_cfg)
        got = next(
            m.local_window for m in eager_model.modules()
            if getattr(m, "local_window", None) is not None
        )
        # The independent schedule above must agree with the trainer's own ramp;
        # if it does not, every row below is mislabelled.
        assert got == want_window, f"schedule mismatch at step {first_step}: want {want_window}, got {got}"

        g0 = graphs()
        for _ in range(args.settle):
            one_step()
        torch.cuda.synchronize()
        recompiles = graphs() - g0

        t0 = time.perf_counter()
        for _ in range(args.iters):
            one_step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tok_s = args.iters * accum * batch * seq / elapsed
        rows.append((first_step, got, tok_s, recompiles, torch.cuda.max_memory_allocated() / 1e9))
        print(f"  step {first_step:6d}  window {got:5d}  {tok_s:10,.0f} tok/s  +{recompiles} graphs", flush=True)

    control = rows[-1][2]  # the final window is the production setting
    print(f"\n{'step':>7} {'window':>7} {'tok/s':>11} {'vs final':>9} {'graphs':>7} {'peak GB':>8}")
    for step, w, tok_s, rc, gb in rows:
        print(f"{step:7d} {w:7d} {tok_s:11,.0f} {tok_s / control:8.3f}x {rc:7d} {gb:8.2f}")

    # What the ramp is actually worth: time-weighted mean throughput over the
    # warmup, against running the whole warmup at the final window. Comparing
    # peak plateau speed to the control would badly oversell it — the ramp
    # spends most of its steps near the top.
    bounds = [s for s, _, _, _, _ in rows] + [args.warmup_steps]
    weighted = sum(
        (bounds[i + 1] - bounds[i]) / rows[i][2] for i in range(len(rows))
    )
    eff = (args.warmup_steps - bounds[0]) / weighted if weighted else float("nan")
    print(f"\ntotal unique graphs: {graphs()} (cache_size_limit {limit})")
    print(f"warmup-average {eff:,.0f} tok/s vs {control:,.0f} at the final window = {eff / control:.3f}x")
    # The whole-run number needs the REAL run length, which this bench config
    # does not have: `training.steps` here is a placeholder (10) that exists only
    # so the YAML parses. Reading it produced a "+2979%" whole-run effect on the
    # first run of this script. Require the number explicitly rather than
    # inventing it from a config field that does not mean what it says.
    if args.total_steps:
        share = args.warmup_steps / args.total_steps
        print(
            f"Over a {args.total_steps}-step run the warmup covers {share:.1%} of steps, so the\n"
            f"whole-run effect is ~{(eff / control - 1) * share * 100:+.2f}%."
        )
    else:
        print(
            f"Pass --total-steps <real run length> for the whole-run effect; the ramp is only\n"
            f"worth {args.warmup_steps} steps of this, so the run-level number is much smaller."
        )
    print("Quality is a separate question: shorter windows change what the model can attend to.")


if __name__ == "__main__":
    main()
