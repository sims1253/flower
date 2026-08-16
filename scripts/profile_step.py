#!/usr/bin/env python3
"""Training-step profiler for the 450M long-context bake-off (S14 measurement task).

This is a *measurement* script — it writes no optimisation code. Its job is to
produce a torch.profiler trace and a ranked breakdown of where training-step time
actually goes for configs/sweep13_450m_longctx_memory.yaml, replacing the S14
kernel-size guesses ("3-10% of total step time") with data.

WHAT IT MEASURES
  A real training step in this config is `gradient_accumulation_steps` (=16)
  microsteps of forward+backward (batch 2 x seq 8192 = 16384 tokens each) followed
  by ONE optimizer step. The earlier "optimizer dominates 58%" finding
  (NEXT_IDEAS.md section 5) came from an accum=1 small config; at accum=16 the
  fwd/bwd work is 16x larger while the optimizer step is unchanged, so that claim
  must be re-measured at the real shape. We measure the full accumulated step.

  We build the model + optimizer EXACTLY as flower/train.py train() does:
    - load the sweep config and merge the per-variant overrides (no re-hardcoding)
    - configure_precision / configure_vram_limit (the VRAM cap guards the run)
    - build_model(...).to(device); build_optimizer(...) on the eager module
    - disable collect_module_diagnostics (untraceable by Dynamo)
    - prebuild_attention_masks (so compiled forward only reads cached masks)
    - torch.compile(mode=cfg.compile_mode, dynamic=False)
    - bf16 autocast on the forward, exactly as the training loop

  Synthetic tokens stand in for fineweb_edu so the profile needs no dataset
  download; kernel time is independent of token *content*, only of shape/dtype.

OUTPUTS (per variant, under docs/profiling/traces/):
  - {variant}.trace.json       Chrome trace for chrome://tracing / Perfetto
  - stdout: KeyAverager tables + a ranked category breakdown + phase timing

USAGE
  uv run python scripts/profile_step.py                       # both arms, 450M
  uv run python scripts/profile_step.py --variant bloom_memory
  uv run python scripts/profile_step.py --warmup 20 --profile-steps 10
  uv run python scripts/profile_step.py --accum 1             # micro-step shape (A/B)

The defaults (warmup=20, profile-steps=10, accum from config) match the
measurement protocol: 20 warmup steps to clear compile + cudagraph capture, then
10 profiled steps.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from flower.config import load_config
from flower.models import build_model
from flower.models.base import count_parameters, prebuild_attention_masks
from flower.optim import build_optimizer
from flower.precision import maybe_convert_fp8
from flower.sweep import load_sweep, select_variants
from flower.train import configure_precision, configure_vram_limit, resolve_device

# --------------------------------------------------------------------------- #
# Op-name -> category attribution.
#
# torch.compile fuses many ops, so the raw op names under a compiled fwd/bwd are
# inductor/triton kernels (triton_poi, triton_mm, cutlass fmha, ...) rather than
# eager aten names. The optimizer runs EAGER (built on the eager module), so its
# ops keep their aten:: names (bmm, mm, norm, ...). We match on a broad set of
# substrings so both compiled and eager ops land in the right bucket. Attribution
# by name is an inherent lower-bound (e.g. `mm` serves FFN, projections, AND the
# optimizer NS), so the category table is a guide; the raw Top-N tables are
# ground truth. Caveats are printed alongside the breakdown.
# --------------------------------------------------------------------------- #
CATEGORIES: dict[str, tuple[str, ...]] = {
    "attention (flex/fmha/sdpa)": (
        "flex_attention", "_scaled_dot_product", "_flash_attention_forward",
        "_efficient_attention_forward", "fmha", "flash", "sdpa",
        "_causal", "attention", "score",
        # Triton flex-attention fused kernels (forward/backward/score/transpose).
        "triton_tem_fused__unsafe_view_flex_attention",
    ),
    "ffn / projections (gemm)": (
        # cuBLAS/cutlass GEMM kernels that the compiled path lowers mm/addmm into.
        "addmm", "mm", "bmm", "linear", "gemm", "sgemm",
        "cutlass_80_tensorop", "cutlass::kernel", "matmul", "_matmul", "triton_mm",
    ),
    "optimizer (Muon NS + AdamW)": (
        # Eager optimizer dispatch (not compiled). The Muon NS matmuls show up as
        # aten::bmm/mm but are categorised here only when the op name itself is
        # optimizer-specific; the bulk is captured via phase timing instead.
        "_newtonschulz", "zeropower", "newton", "schulz", "_foreach_norm",
        "_foreach_mul_", "_foreach_add_",
    ),
    "norm / activation / elementwise": (
        "layer_norm", "rms_norm", "rmsnorm", "_softmax", "softmax",
        "nll_loss", "nll_loss_forward", "cross_entropy",
        "log_softmax", "mul", "add", "sub", "silu", "gelu", "tanh",
        "rsqrt", "pow", "sum", "mean", "clamp", "where", "copy", "clone",
        "unsqueeze", "expand", "reshape", "cat", "slice", "select",
        "to", "cast", "convert", "bitcast", "multi_tensor_apply",
    ),
    "lm_head + cross-entropy": (
        "cross_entropy", "nll_loss", "nll_loss_forward", "log_softmax",
        "lm_head",
    ),
    "data / embedding / gather": (
        "embedding", "gather", "index", "arange", "_to_copy",
    ),
    "compiled region (un-attributed)": (
        # Under cudagraph mode most GPU time collapses into the monolithic
        # compiled-region wrapper rather than named kernels. This bucket makes
        # that visible rather than hiding it in "other".
        "torch-compiled region", "compiled region",
    ),
}


def _is_inclusive_wrapper(name: str) -> bool:
    """True for profiler entries that are INCLUSIVE parents of other ops.

    Under torch.compile + cudagraph, `## Call CompiledFxGraph ...` reports its
    self-CUDA time as the sum of every kernel the compiled region replays, and
    those same kernels ALSO appear as their own key_averages entries. Summing
    self-CUDA across everything double/triple-counts (a 400 ms wall step reports
    seconds of "CUDA time"). These wrapper entries must be excluded from any
    sum-over-ops; they're useful in the Top-N table (they show which compiled
    graph dominates) but never in a total.
    """
    return (
        name.startswith("## Call CompiledFxGraph")
        or name.startswith("CompiledFunctionBackward")
        or name.startswith("autograd::engine::evaluate_function")
        or name.startswith("Optimizer.step#")
        or name.startswith("Optimizer.step")
        or "Command Buffer" in name
    )


def _categorise(name: str) -> str:
    """Return the category for an op name, or 'other'."""
    low = name.lower()
    # Order matters: check the most specific buckets first so e.g. the CE op
    # doesn't get swallowed by the generic elementwise bucket.
    for cat in ("lm_head + cross-entropy", "attention (flex/fmha/sdpa)",
                "optimizer (Muon NS + AdamW)", "data / embedding / gather",
                "ffn / projections (gemm)", "norm / activation / elementwise",
                "compiled region (un-attributed)"):
        if any(s in low for s in CATEGORIES[cat]):
            return cat
    return "other"


def attribute_key_averages(key_averages) -> dict[str, dict[str, float]]:
    """Aggregate profiler KeyAveragers into {category: {cuda_ms, cpu_ms, count}}.

    Uses SELF device time (`e.device_time`), never `device_time_total` — the
    latter is inclusive and double-counts under nested op dispatch. INCLUSIVE
    wrapper entries (compiled-graph replays, optimizer-step wrappers) are skipped
    entirely: their self-CUDA time is the sum of their children, so including
    them would double-count. An op is placed in the FIRST matching category
    (precedence order above), so categories are mutually exclusive and sum to
    <= the wrapper-excluded self-CUDA total. The remainder lands in 'other'.
    """
    out: dict[str, dict[str, float]] = {
        cat: {"cuda_ms": 0.0, "cpu_ms": 0.0, "count": 0.0} for cat in CATEGORIES
    }
    out["other"] = {"cuda_ms": 0.0, "cpu_ms": 0.0, "count": 0.0}
    for e in key_averages:
        if _is_inclusive_wrapper(e.key):
            continue
        cat = _categorise(e.key)
        out[cat]["cuda_ms"] += e.device_time / 1e3          # us -> ms (SELF)
        out[cat]["cpu_ms"] += e.self_cpu_time_total / 1e3   # us -> ms
        out[cat]["count"] += e.count
    return out


# --------------------------------------------------------------------------- #
# Config loading: merge the sweep variant exactly as flower.sweep does, then feed
# the merged dict through the SAME load_config path train.py uses.
# --------------------------------------------------------------------------- #
def load_variant_config(sweep_path: str, variant_name: str, *, accum_override: int | None) -> Any:
    """Return an ExperimentConfig for `variant_name` from a sweep YAML.

    Writes the merged variant config to a temp file and loads it via
    `flower.config.load_config`, so the parsing is identical to a real run. The
    dataset is forced to `synthetic` (same seq/batch) so no download is needed;
    this does not change kernel time.
    """
    _sweep_name, variants = load_sweep(sweep_path)
    selected = select_variants(variants, variant_name, limit=None)
    if not selected:
        raise ValueError(f"Variant {variant_name!r} not found in {sweep_path}")
    merged = selected[0]["config"]
    # Force synthetic data for the compute profile (same shape/dtype).
    merged.setdefault("data", {})["dataset"] = "synthetic"
    merged["data"]["synthetic_vocab_size"] = merged["model"]["vocab_size"]
    merged["data"]["sequence_length"] = merged["data"].get(
        "sequence_length", merged["model"].get("max_seq_len", 8192)
    )
    if accum_override is not None:
        merged.setdefault("training", {})["gradient_accumulation_steps"] = accum_override
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        import yaml

        yaml.safe_dump(merged, tf, sort_keys=False)
        tmp_path = tf.name
    return load_config(tmp_path)


def build_everything(cfg):
    """Mirror flower/train.py train() model+optim+compile wiring exactly."""
    device = resolve_device(cfg.training.device)
    # Honour the config's vram_fraction exactly as train.py does. Hardcoding the
    # 0.85 default here made the harness cap VRAM lower than the run it claims to
    # mirror (the 450M configs set 0.95 for the validation spike), so arms that
    # fit in a real run would OOM under measurement.
    configure_vram_limit(device, fraction=getattr(cfg.training, "vram_fraction", 0.85))
    amp_dtype = configure_precision(cfg.training.precision, device)
    eager_model = build_model(cfg.model).to(device)
    # Same FP8 gate the trainer uses, in the same position (before the optimizer
    # is built, since conversion rebinds weight Parameters). Shared entry point
    # so a profile can never measure a different precision layout than a run.
    eager_model, _fp8_info = maybe_convert_fp8(eager_model, cfg.training, device)
    optim_or_list = build_optimizer(eager_model, cfg.training)
    optims = optim_or_list if isinstance(optim_or_list, list) else [optim_or_list]
    # Match train.py compile-time wiring.
    if cfg.training.compile_model:
        if getattr(eager_model, "collect_module_diagnostics", False):
            eager_model.collect_module_diagnostics = False
        if device.type == "cuda":
            prebuild_attention_masks(eager_model, cfg.data.sequence_length, device)
        run_model = torch.compile(eager_model, mode=cfg.training.compile_mode, dynamic=False)
    else:
        run_model = eager_model
    return device, amp_dtype, eager_model, run_model, optims


def make_synthetic_batch(batch: int, seq: int, vocab: int, device: torch.device):
    """A labelled next-token batch: labels = ids (the model shifts internally)."""
    ids = torch.randint(0, vocab, (batch, seq), device=device, dtype=torch.long)
    return ids, ids


def profile_variant(
    sweep_path: str,
    variant_name: str,
    *,
    warmup: int,
    profile_steps: int,
    accum_override: int | None,
    trace_path: Path | None,
) -> dict[str, Any]:
    torch.manual_seed(0)
    cfg = load_variant_config(sweep_path, variant_name, accum_override=accum_override)
    accum = max(1, int(cfg.training.gradient_accumulation_steps))
    batch = int(cfg.training.batch_size)
    seq = int(cfg.data.sequence_length)
    vocab = int(cfg.model.vocab_size)
    device, amp_dtype, eager_model, run_model, optims = build_everything(cfg)
    params = count_parameters(eager_model)
    run_model.train()

    def one_microstep():
        ids, labels = make_synthetic_batch(batch, seq, vocab, device)
        with torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else _nullctx():
            out = run_model(ids, labels=labels)
            loss = out["loss"]
        loss.backward()

    def one_full_step():
        for o in optims:
            o.zero_grad(set_to_none=True)
        for _ in range(accum):
            one_microstep()
        # grad clip + optim step, exactly as train.py orders them.
        torch.nn.utils.clip_grad_norm_(eager_model.parameters(), cfg.training.grad_clip)
        for o in optims:
            o.step()

    # --- compile + cudagraph warmup -------------------------------------------
    compile_t0 = time.perf_counter()
    for _ in range(warmup):
        one_full_step()
    torch.cuda.synchronize()
    compile_secs = time.perf_counter() - compile_t0

    # --- wall-clock throughput (un-profiled, clean) ---------------------------
    torch.cuda.reset_peak_memory_stats()
    wall_t0 = time.perf_counter()
    for _ in range(profile_steps):
        one_full_step()
    torch.cuda.synchronize()
    wall_secs = time.perf_counter() - wall_t0
    ms_per_step = wall_secs / profile_steps * 1e3
    tok_s = profile_steps * accum * batch * seq / wall_secs
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # --- torch.profiler (MUST run before any other profiler context) ----------
    # IMPORTANT ordering constraint: on torch 2.13/cu130, a prior
    # `torch.profiler.profile(...)` context (even CPU-only) corrupts the CUDA
    # activity channel so the NEXT profile records 0 device time. Empirically
    # verified: launch-count passes run before the main profile -> main profile
    # gets device_time == 0 for every op. So the main CUDA profile runs FIRST,
    # immediately after the un-profiled warmup/wall-clock block above.
    #
    # with_stack=False: with_stack=True makes key_averages() aggregate by
    # (op, call-stack), so the same kernel appears under every stack frame that
    # touches it and summing device_time double/triple-counts (a 400 ms step
    # reports 21 SECONDS of "CUDA time"). with_stack=False gives correct self
    # CUDA time; the Chrome trace still has full op + shape detail for
    # chrome://tracing / Perfetto, just without Python stack frames.
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    prof_wall_t0 = time.perf_counter()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(profile_steps):
            one_full_step()
    torch.cuda.synchronize()
    prof_wall_secs = time.perf_counter() - prof_wall_t0
    prof_ms_per_step = prof_wall_secs / profile_steps * 1e3

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))

    ka = prof.key_averages()
    # device_time = SELF CUDA time (excludes children). device_time_total is
    # inclusive and double-counts under nested op dispatch; never sum it. We also
    # exclude the inclusive wrapper entries (see _is_inclusive_wrapper) whose
    # self-CUDA time is the sum of their children.
    leaf_entries = [e for e in ka if not _is_inclusive_wrapper(e.key)]
    total_cuda_ms = sum(e.device_time for e in leaf_entries) / 1e3 / profile_steps
    total_cpu_ms = sum(e.self_cpu_time_total for e in ka) / 1e3 / profile_steps
    total_kernels = sum(e.count for e in ka) / profile_steps
    cat = attribute_key_averages(ka)
    cat = {k: {kk: vv / profile_steps for kk, vv in v.items()} for k, v in cat.items()}

    # --- phase timing (wall-clock with explicit cuda.synchronize) -------------
    # The clean fwd / bwd / opt split. CUDA-event timing mis-attributes the eager
    # optimizer (its kernels are launched async after the event records), and the
    # compiled-graph wrapper double-counts in the profiler. Wall-clock between
    # synchronized phase boundaries is unambiguous: it includes both GPU compute
    # AND host dispatch / launch overhead, which is exactly what determines step
    # wall-time. The "idle gap" the task asks about surfaces as the difference
    # between (fwd+bwd+opt) and the full-step wall-clock.
    def _timed(fn, reps: int = 5) -> float:
        for _ in range(2):  # warm
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    def _do_fwd():
        ids, labels = make_synthetic_batch(batch, seq, vocab, device)
        with torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else _nullctx():
            run_model(ids, labels=labels)["loss"]

    fwd_ms = _timed(_do_fwd) * accum  # one microstep * accum = full-step fwd total

    def _do_fwbwd_micro():
        for o in optims:
            o.zero_grad(set_to_none=True)
        ids, labels = make_synthetic_batch(batch, seq, vocab, device)
        with torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else _nullctx():
            loss = run_model(ids, labels=labels)["loss"]
        loss.backward()

    fwbwd_micro_ms = _timed(_do_fwbwd_micro)
    bwd_ms = (fwbwd_micro_ms - fwd_ms / accum) * accum  # bwd-per-micro * accum

    def _do_full_step_no_sync():
        for o in optims:
            o.zero_grad(set_to_none=True)
        for _ in range(accum):
            ids, labels = make_synthetic_batch(batch, seq, vocab, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else _nullctx():
                loss = run_model(ids, labels=labels)["loss"]
            loss.backward()
        torch.nn.utils.clip_grad_norm_(eager_model.parameters(), cfg.training.grad_clip)
        for o in optims:
            o.step()

    full_step_ms = _timed(_do_full_step_no_sync)

    # opt_ms from the difference: (fwd+bwd+clip+opt) full step vs (fwd+bwd) only.
    # Timing opt.step() alone on stale grads would under-measure (no grad-clip
    # dispatch and no inter-phase launch overhead), so derive from the full step
    # instead. fwd_ms+bwd_ms already include their per-microstep launch overhead;
    # the residual is opt + grad-clip + inter-phase dispatch — the quantity that
    # determines whether the optimizer is worth optimising.
    opt_ms = max(0.0, full_step_ms - (fwd_ms + bwd_ms))
    phase = {"fwd": fwd_ms, "bwd": bwd_ms, "opt": opt_ms,
             "full_step_syncd": full_step_ms}

    # --- per-phase kernel launch counts (run LAST; CPU-only profiles) ---------
    # Compiled triton kernels share names across fwd/bwd, so launch counts cannot
    # be split from the combined profile. Run each phase in isolation under a
    # CPU-only profile (no CUDA activity, no shapes -> no extra GPU memory beyond
    # the phase's own graph; peak ~17 GB for a single fwd+bwd vs the 27 GB cap).
    # These run AFTER the main CUDA profile because of the ordering constraint
    # noted above (a prior profiler context zeroes the next one's CUDA channel).
    def _launch_count(fn, reps: int = 2) -> int:
        for _ in range(reps):  # warm
            fn()
        torch.cuda.empty_cache()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
            for _ in range(reps):
                fn()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return sum(e.count for e in p.key_averages()) // reps

    def _do_fwd_only():
        ids, labels = make_synthetic_batch(batch, seq, vocab, device)
        for o in optims:
            o.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else _nullctx():
            run_model(ids, labels=labels)["loss"]

    def _do_fwbwd():
        for o in optims:
            o.zero_grad(set_to_none=True)
        ids, labels = make_synthetic_batch(batch, seq, vocab, device)
        with torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else _nullctx():
            loss = run_model(ids, labels=labels)["loss"]
        loss.backward()

    def _do_opt():
        for o in optims:
            o.step()

    fwd_kernels = _launch_count(_do_fwd_only)
    fwbwd_kernels = _launch_count(_do_fwbwd)
    opt_kernels = _launch_count(_do_opt)
    bwd_kernels = max(0, fwbwd_kernels - fwd_kernels)

    return {
        "variant": variant_name,
        "model_variant": cfg.model.variant,
        "params": params,
        "accum": accum,
        "batch": batch,
        "seq": seq,
        "vocab": vocab,
        "precision": cfg.training.precision,
        "compiled": cfg.training.compile_model,
        "warmup_secs": compile_secs,
        "ms_per_step": ms_per_step,
        "tok_s": tok_s,
        "peak_gb": peak_gb,
        "prof_ms_per_step": prof_ms_per_step,
        "profiler_cuda_ms_per_step": total_cuda_ms,
        "profiler_cpu_ms_per_step": total_cpu_ms,
        "total_kernels_per_step": total_kernels,
        # GPU util = busy CUDA time / wall time OF THE PROFILED STEPS. Profiling
        # slows the profiled steps vs the clean wall-clock above, so the honest
        # denominator is the profiled wall time, not ms_per_step. Still an upper
        # bound: self-CUDA-time summed across ops can overlap with launch gaps,
        # so >100% means "GPU-bound within the profiled window".
        "gpu_util_pct": 100.0 * total_cuda_ms / prof_ms_per_step if prof_ms_per_step > 0 else 0.0,
        "phase_ms": phase,
        "categories": cat,
        "kernel_counts": {
            "forward": fwd_kernels,
            "backward": bwd_kernels,
            "optimizer": opt_kernels,
            "total": fwd_kernels + bwd_kernels + opt_kernels,
        },
        "top_cuda_table": ka.table(sort_by="self_cuda_time_total", row_limit=20,
                                   max_name_column_width=55),
        "top_cpu_table": ka.table(sort_by="self_cpu_time_total", row_limit=20,
                                  max_name_column_width=55),
    }


def _nullctx():
    return contextlib.nullcontext()


def print_report(r: dict[str, Any]) -> None:
    accum = r["accum"]
    print(f"\n{'=' * 90}")
    print(f"{r['variant']}  [{r['model_variant']}]  {r['params']/1e6:.0f}M params  "
          f"seq={r['seq']} batch={r['batch']} accum={accum}  "
          f"{r['precision']} compiled={r['compiled']}")
    print(f"{'=' * 90}")
    eff_batch = accum * r["batch"] * r["seq"]
    print(f"effective batch:       {eff_batch:,} tokens/step ({accum} x {r['batch']} x {r['seq']})")
    print(f"wall-clock:            {r['ms_per_step']:8.2f} ms/step   {r['tok_s']:,.0f} tok/s   peak {r['peak_gb']:.2f} GB")
    print(f"  warmup (compile+cgraph): {r['warmup_secs']:6.1f}s for the configured warmup block")
    print(f"  profiled wall:       {r['prof_ms_per_step']:8.2f} ms/step (profiler overhead inflates this vs the clean wall-clock above)")
    print(f"profiler CUDA total:   {r['profiler_cuda_ms_per_step']:8.2f} ms/step   "
          f"(GPU util {r['gpu_util_pct']:.1f}% = cuda_ms / profiled_wall; >100% = GPU-bound)")
    print(f"profiler CPU total:    {r['profiler_cpu_ms_per_step']:8.2f} ms/step")
    print(f"kernels/step:          {r['kernel_counts']['total']} "
          f"[fwd {r['kernel_counts']['forward']} + bwd {r['kernel_counts']['backward']} "
          f"+ opt {r['kernel_counts']['optimizer']}]")

    print("\nPhase timing (wall-clock between synchronised phase boundaries, ms):")
    # Phase keys: 'fwd','bwd' are summed across all accum microsteps (each timed
    # alone, ×accum); 'opt' = full_step − (fwd+bwd), i.e. optimizer step + grad-
    # clip + inter-phase dispatch. The task asks "is there a long idle gap between
    # backward and optimizer?" — that gap is folded into 'opt' here; if it were
    # large, 'opt' would dwarf the known optimizer cost (~2.4% for bloom, ~0% for
    # vanilla), so a small 'opt' == no idle gap.
    ph = r["phase_ms"]
    fwd, bwd = ph.get("fwd", 0.0), ph.get("bwd", 0.0)
    opt = ph.get("opt", 0.0)             # optimizer + grad-clip + inter-phase dispatch
    full = ph.get("full_step_syncd", r["ms_per_step"])
    # The fwd+bwd+opt split uses synchronised wall-clock; the un-syncd ms_per_step
    # is the real training-loop number. fwd+bwd can slightly exceed it because the
    # phases are measured in separate timed runs (noise); the residual is small.
    print(f"  {'fwd (all microsteps)':28s} {fwd:8.2f} ms  ({100*fwd/r['ms_per_step']:5.1f}% of wall step)")
    print(f"  {'bwd (all microsteps)':28s} {bwd:8.2f} ms  ({100*bwd/r['ms_per_step']:5.1f}% of wall step)")
    print(f"  {'optimizer+clip+gap':28s} {opt:8.2f} ms  ({100*opt/r['ms_per_step']:5.1f}% of wall step)  <- Muon NS + AdamW + dispatch")
    print(f"  {'(syncd full step, ref)':28s} {full:8.2f} ms")

    cuda_total = r["profiler_cuda_ms_per_step"]
    print("\nCategory breakdown (CUDA time, mutually exclusive, per step):")
    print(f"  {'category':38s} {'cuda ms':>9s} {'%cuda':>7s} {'%step':>7s} {'kernels':>9s}")
    rows = sorted(r["categories"].items(), key=lambda kv: -kv[1]["cuda_ms"])
    for cat, v in rows:
        pct_cuda = 100.0 * v["cuda_ms"] / cuda_total if cuda_total > 0 else 0.0
        pct_step = 100.0 * v["cuda_ms"] / r["ms_per_step"]
        print(f"  {cat:38s} {v['cuda_ms']:9.2f} {pct_cuda:6.1f}% {pct_step:6.1f}% {v['count']:9.0f}")

    print("\nTop 20 CUDA ops by self device time (per step avg):")
    print(r["top_cuda_table"])
    print("\nTop 20 CPU ops by self CPU time (per step avg):")
    print(r["top_cpu_table"])
    print("Note: category attribution by op-name substring is a lower bound — `mm`/`bmm`")
    print("serve FFN, projections AND the optimizer NS. The phase timing above is the")
    print("clean fwd/bwd/opt split; the Top-N tables are ground truth.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/sweep13_450m_longctx_memory.yaml")
    p.add_argument("--variant", default=None,
                   help="sweep variant name; default = run both bake-off arms")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--profile-steps", type=int, default=10)
    p.add_argument("--accum", type=int, default=None,
                   help="override gradient_accumulation_steps (default: from config = 16)")
    p.add_argument("--trace-dir", default="docs/profiling/traces")
    args = p.parse_args()

    if args.variant:
        targets = [args.variant]
    else:
        # The bake-off pair: param-matched vanilla control + bloom memory arm.
        _name, variants = load_sweep(args.config)
        targets = [v["name"] for v in select_variants(variants, None, limit=None)]

    results: list[dict[str, Any]] = []
    for vname in targets:
        trace_path = Path(args.trace_dir) / f"{vname}.trace.json"
        r = profile_variant(
            args.config, vname,
            warmup=args.warmup, profile_steps=args.profile_steps,
            accum_override=args.accum, trace_path=trace_path,
        )
        r["trace_path"] = str(trace_path)
        print_report(r)
        results.append(r)
        # Reset Dynamo between variants so the second variant compiles cleanly.
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    if len(results) > 1:
        print(f"\n{'=' * 90}")
        print("SIDE-BY-SIDE (per full step; accum from config)")
        print(f"{'=' * 90}")
        hdr = f"{'variant':22s} {'ms/step':>9s} {'tok/s':>10s} {'peakGB':>8s} {'GPUutil':>8s} {'opt%':>6s} {'fwd+bwd%':>9s}"
        print(hdr)
        for r in results:
            # Derive opt vs fwd+bwd from phase timing (syncd wall-clock ms).
            opt_ms = r["phase_ms"].get("opt", 0.0)
            fwbwd_ms = r["phase_ms"].get("fwd", 0.0) + r["phase_ms"].get("bwd", 0.0)
            opt_pct = 100.0 * opt_ms / r["ms_per_step"]
            fb_pct = 100.0 * fwbwd_ms / r["ms_per_step"]
            print(f"{r['variant']:22s} {r['ms_per_step']:9.2f} {r['tok_s']:10,.0f} "
                  f"{r['peak_gb']:8.2f} {r['gpu_util_pct']:7.1f}% {opt_pct:5.1f}% {fb_pct:8.1f}%")

    # Drop a machine-readable summary next to the traces for the writeup step.
    summary_path = Path(args.trace_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_jsonify(results), indent=2, sort_keys=True))
    print(f"\n[summary] wrote {summary_path}")


def _jsonify(obj: Any) -> Any:
    """Make profiler results JSON-serialisable (Path -> str, tables stay str)."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


if __name__ == "__main__":
    main()
