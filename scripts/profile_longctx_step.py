#!/usr/bin/env python3
"""Long-context training-step profiler for the sweep13 bake-off arms.

Complements ``scripts/profile_bloom_step.py``: that one profiles the small
(seq=512, full-attention) bloom config to isolate the bloom machinery. This one
profiles the *actual long-context bake-off* shape — seq=8192, window=2048,
flex_attention, torch.compile — across all three arms (vanilla_local,
bloom_memory, summary_memory) so we can see where time goes at the config that
actually runs overnight, and whether the SDPCrossAttention path is on the
hotpath.

Reports, per variant:
  - wall-clock ms/step and tok/s
  - peak GPU memory
  - the top CUDA ops by device time
  - a coarse per-component attribution: local self-attention (flex /
    _scaled_dot_product), memory cross-attention (SDPCrossAttention's
    matmuls + the _eflash/flash path), FFN, and the optimizer step

The optimizer is profiled separately because NEXT_IDEAS.md section 5 already
showed Muon dominates the small-config step; this confirms whether that holds at
long context too.

  uv run python scripts/profile_longctx_step.py [--variant summary_memory] [--seq 8192] [--batch 2]
  uv run python scripts/profile_longctx_step.py --all          # all 3 arms, side by side

Defaults match configs/sweep13_longctx_memory_bakeoff.yaml (d512/L8, seq8192,
window2048, flex, compile, bf16, muon) except batch defaults to 2 so the
profile runs fast; pass --batch 8 for the real sweep shape.
"""

from __future__ import annotations

import argparse
import time

import torch

from flower.config import ModelConfig, TrainingConfig
from flower.models import build_model
from flower.optim import build_optimizer


def make_cfg(variant: str, *, seq: int, batch: int) -> ModelConfig:
    # Mirror configs/sweep13_longctx_memory_bakeoff.yaml defaults.
    base = dict(
        variant=variant,
        vocab_size=16384,
        d_model=512,
        num_heads=8,
        num_layers=8,
        ffn_dim=2048,
        max_seq_len=seq,
        local_window=2048,
        rope_base=10000.0,
        dropout=0.0,
        norm_type="rmsnorm",
        ffn_activation="swiglu",
        ffn_param_match=True,
        qk_norm=True,
        use_bias=False,
        init_scheme="scaled",
        init_std=0.02,
        flex_attention=True,
        memory_slots=8,
    )
    if variant == "bloom_memory":
        base.update(bloom_num_hashes=4, bloom_summary_points=16)
    if variant == "summary_memory":
        base.update(summary_style="perceiver")
    return ModelConfig(**base)


# Op-name substrings used to coarsely attribute CUDA time to a component. The
# attribution is a lower bound (op names overlap; e.g. addmm/mm serve both FFN
# and projections) but it's enough to see *relative* weight.
COMPONENT_OPS = {
    "flex / SDPA (local self-attn)": {
        "flex_attention",
        "_scaled_dot_product",
        "_flash_attention_forward",
        "_efficient_attention_forward",
    },
    "matmul (FFN + projections)": {"addmm", "mm", "bmm", "linear"},
    "elementwise / norm / softmax": {
        "layer_norm",
        "rms_norm",
        "_softmax",
        "softmax",
        "mul",
        "add",
        "silu",
        "gelu",
        "tanh",
        "rsqrt",
        "pow",
    },
    "optimizer (Muon NS + AdamW)": {
        "_newtonschulz5",  # internal name may vary; keep for matching
        "_zeropower",
        "norm",
    },
}


def attribute(prof) -> dict[str, tuple[float, int]]:
    """Return {component: (cuda_ms, op_count)} by substring-matching op names."""
    out: dict[str, tuple[float, int]] = {k: (0.0, 0) for k in COMPONENT_OPS}
    for e in prof.key_averages():
        if e.device_time_total <= 0:
            continue
        key_us = e.device_time_total
        name = e.key.lower()
        for comp, substrings in COMPONENT_OPS.items():
            if any(s in name for s in substrings):
                ms, n = out[comp]
                out[comp] = (ms + key_us / 1e3, n + e.count)
                break
    return out


def profile_variant(variant: str, *, seq: int, batch: int, warmup: int, steps: int, profile_steps: int) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(0)
    cfg = make_cfg(variant, seq=seq, batch=batch)
    model = build_model(cfg).to(device).to(torch.bfloat16).train()
    model.collect_module_diagnostics = False  # off under compile (S14 Opp. 4)
    tcfg = TrainingConfig(
        batch_size=batch, steps=warmup + steps, device="cuda", precision="bf16",
        optimizer="muon", muon_lr=0.003, muon_momentum=0.95,
        weight_decay=0.01, weight_decay_exclude_embeddings=True,
    )
    optims = build_optimizer(model, tcfg)
    optims = optims if isinstance(optims, list) else [optims]
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    compiled = torch.compile(model, mode="default", dynamic=False)
    vocab = cfg.vocab_size

    def one_step():
        for o in optims:
            o.zero_grad(set_to_none=True)
        ids = torch.randint(0, vocab, (batch, seq), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = compiled(ids, labels=ids)
        out["loss"].backward()
        for o in optims:
            o.step()

    # Warmup (also triggers cudagraph capture + inductor compile).
    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms_per_step = elapsed / steps * 1e3
    tok_s = steps * batch * seq / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # Profile a few representative steps.
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        for _ in range(profile_steps):
            one_step()
    torch.cuda.synchronize()

    total_cuda = sum(e.device_time_total for e in prof.key_averages()) / 1e3 / profile_steps  # ms/step
    attrib = attribute(prof)
    # attribute() summed over profile_steps; normalise to per-step.
    attrib = {k: (v[0] / profile_steps, v[1] / profile_steps) for k, v in attrib.items()}

    # Fwd+bwd vs optimizer split: time the same step with and without opt.step().
    # Quick A/B: a single timed step each, after a synchronize.
    torch.cuda.synchronize()
    ids = torch.randint(0, vocab, (batch, seq), device=device)

    def fwd_bwd_only():
        for o in optims:
            o.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = compiled(ids, labels=ids)
        out["loss"].backward()

    # warm the path
    fwd_bwd_only()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fwd_bwd_only()
    torch.cuda.synchronize()
    fwd_bwd_ms = (time.perf_counter() - t0) * 1e3

    return {
        "variant": variant,
        "params_M": params_M,
        "ms_per_step": ms_per_step,
        "tok_s": tok_s,
        "peak_gb": peak_gb,
        "profiler_cuda_ms_per_step": total_cuda,
        "fwd_bwd_ms": fwd_bwd_ms,
        "opt_ms_estimate": ms_per_step - fwd_bwd_ms,
        "attribution": attrib,
        "top_ops_table": prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=12, max_name_column_width=42
        ),
    }


def print_report(r: dict) -> None:
    print(f"\n{'='*78}")
    print(f"{r['variant']}  d512/L8 seq=8192 window=2048 flex+compile bf16 muon  ({r['params_M']:.1f}M params)")
    print(f"{'='*78}")
    print(f"wall-clock:        {r['ms_per_step']:7.2f} ms/step   {r['tok_s']:,.0f} tok/s   peak {r['peak_gb']:.2f} GB")
    print(f"  fwd+bwd (timed): {r['fwd_bwd_ms']:7.2f} ms")
    print(f"  optimizer (diff):{r['opt_ms_estimate']:7.2f} ms  ({100*r['opt_ms_estimate']/r['ms_per_step']:.0f}% of step)")
    print(f"profiler CUDA:     {r['profiler_cuda_ms_per_step']:7.2f} ms/step (fwd+bwd+opt, normalised)")
    print(f"\nCoarse CUDA attribution (per step, lower-bound):")
    for comp, (ms, n) in sorted(r["attribution"].items(), key=lambda kv: -kv[1][0]):
        pct = 100 * ms / r["profiler_cuda_ms_per_step"] if r["profiler_cuda_ms_per_step"] > 0 else 0
        print(f"  {comp:34s} {ms:7.2f} ms  ({pct:4.1f}%)  [{n:.0f} ops]")
    print(f"\nTop 12 CUDA ops by total device time (summed over profiled steps):")
    print(r["top_ops_table"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="summary_memory", choices=["vanilla_local", "bloom_memory", "summary_memory"])
    p.add_argument("--all", action="store_true", help="profile all 3 arms")
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--profile-steps", type=int, default=3)
    args = p.parse_args()

    variants = ["vanilla_local", "bloom_memory", "summary_memory"] if args.all else [args.variant]
    results = []
    for v in variants:
        r = profile_variant(v, seq=args.seq, batch=args.batch, warmup=args.warmup, steps=args.steps, profile_steps=args.profile_steps)
        results.append(r)
        print_report(r)
        torch._dynamo.reset()

    if len(results) > 1:
        print(f"\n{'='*78}")
        print("SUMMARY (wall-clock ms/step, tok/s, peak GB)")
        print(f"{'='*78}")
        print(f"{'variant':20s} {'ms/step':>9s} {'tok/s':>10s} {'peak GB':>9s} {'fwd+bwd':>9s} {'opt(ms)':>9s}")
        for r in results:
            print(f"{r['variant']:20s} {r['ms_per_step']:9.2f} {r['tok_s']:10,.0f} {r['peak_gb']:9.2f} {r['fwd_bwd_ms']:9.2f} {r['opt_ms_estimate']:9.2f}")


if __name__ == "__main__":
    main()
