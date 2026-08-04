#!/usr/bin/env python3
"""Full training-step profiler for bloom_memory.

S14 Opportunity 2 Part A was about collapsing the K-hash loop. This script
profiles the *whole* forward+backward+optimizer step (not just _bloom_route)
and breaks down CUDA time by operation, so we can see what's actually on the
hot path and whether anything in the bloom memory path is worth touching next.

Runs eager (default) and --compile (matches how real sweep runs train). Reports
the top CUDA-time ops and a focused breakdown of the bloom-memory-specific ops
(the hash einsum, softmax, perceiver summary attn, write-value MLP, mem_read).

  uv run python scripts/profile_bloom_step.py [--compile] [--layers 4] [--d-model 384]
"""

from __future__ import annotations

import argparse
import time

import torch

from flower.config import ModelConfig, TrainingConfig
from flower.models import build_model
from flower.optim import build_optimizer


def make_cfg(d_model: int, layers: int, seq: int, batch: int) -> ModelConfig:
    return ModelConfig(
        variant="bloom_memory",
        vocab_size=256,
        d_model=d_model,
        num_heads=d_model // 64,
        num_layers=layers,
        ffn_dim=d_model * 4,
        max_seq_len=seq,
        local_window=seq,  # full attention, no flex — isolate compute
        memory_slots=16,
        bloom_num_hashes=4,
        bloom_summary_points=16,
    )


def bloom_cuda_total(prof) -> float:
    """Sum CUDA time of ops attributable to the bloom memory write/read path."""
    bloom_keys = {
        "aten::einsum",        # hash matmul
        "aten::softmax",       # hash routing softmax
        "aten::_softmax",
        "aten::mean",          # plan = stacked.mean
        "aten::bmm",           # per_slot_write = plan.T @ values ; mem_read attn
        "aten::clamp_min",     # diagnostics
        "aten::log",           # diagnostics
        "aten::sum",           # diagnostics
    }
    total = 0.0
    counts = {}
    for e in prof.key_averages():
        if e.key in bloom_keys and e.device_time_total > 0:
            total += e.device_time_total
            counts[e.key] = counts.get(e.key, 0) + e.count
    return total, counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--optimizer", type=str, default="muon")
    args = p.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)
    cfg = make_cfg(args.d_model, args.layers, args.seq, args.batch)
    model = build_model(cfg).to(device).train()
    tcfg = TrainingConfig(batch_size=args.batch, steps=args.warmup + args.steps,
                          device="cuda", precision="bf16", optimizer=args.optimizer,
                          muon_lr=0.02)
    optims = build_optimizer(model, tcfg)
    optims = optims if isinstance(optims, list) else [optims]
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    run = model
    if args.compile:
        if getattr(model, "collect_module_diagnostics", False):
            model.collect_module_diagnostics = False
        run = torch.compile(model, dynamic=False)

    amp = torch.bfloat16
    vocab = cfg.vocab_size

    def one_step():
        for o in optims:
            o.zero_grad(set_to_none=True)
        ids = torch.randint(0, vocab, (args.batch, args.seq), device=device)
        with torch.amp.autocast("cuda", dtype=amp):
            out = run(ids, labels=ids)
        out["loss"].backward()
        for o in optims:
            o.step()

    # Warmup
    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize()

    # Wall-clock throughput
    start = time.perf_counter()
    for _ in range(args.steps):
        one_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms_per_step = elapsed / args.steps * 1e3
    tok_s = args.steps * args.batch * args.seq / elapsed

    # Profile one representative step
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        one_step()
    torch.cuda.synchronize()

    total_cuda = sum(e.device_time_total for e in prof.key_averages()) / 1e3  # ms
    bloom_us, bloom_counts = bloom_cuda_total(prof)
    bloom_ms = bloom_us / 1e3

    tag = " (compiled)" if args.compile else ""
    print(f"\n{'='*72}")
    print(f"bloom_memory{tag}  d={cfg.d_model} L={cfg.num_layers} seq={args.seq} B={args.batch} "
          f"opt={args.optimizer}  ({params_M:.1f}M params)")
    print(f"{'='*72}")
    print(f"wall-clock:  {ms_per_step:.2f} ms/step   {tok_s:,.0f} tok/s")
    print(f"profiler CUDA total (1 step, fwd+bwd+opt): {total_cuda:.2f} ms")
    print(f"bloom-path CUDA (subset, see note):        {bloom_ms:.2f} ms "
          f"({100*bloom_ms/total_cuda:.1f}% of profiled CUDA)")
    print(f"  bloom ops counted: {dict(bloom_counts)}")
    print(f"\nTop 15 CUDA ops by total device time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15,
                                    max_name_column_width=45))
    print("Note: bloom-path CUDA is a *lower bound* — bmm/einsum also serve the")
    print("attention and FFN matmuls which dominate; this counts the bloom-routed")
    print("subset of those op names. Use the per-op table for the real picture.")


if __name__ == "__main__":
    main()
