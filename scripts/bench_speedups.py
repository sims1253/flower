#!/usr/bin/env python3
"""Benchmark training-speedups.md features: legacy vs. safe throughput vs. full.

Measures steady-state forward+backward+step throughput and peak VRAM for three
configurations at IDENTICAL total compute (same model, batch, sequence), so the
tok/s comparison is fair. Synthetic tokens isolate compute from data loading.

  uv run python scripts/bench_speedups.py

The three arms:
  legacy      — BF16 + Muon (the project's current research baseline).
  throughput  — + FlexAttention, FP8 lm_head (eval), BF16 CE.
  full        — + NorMuon, Cautious WD, Smooth-SwiGLU, orthogonal init.

FlexAttention's win only shows at long sequence (it eliminates the dense T x T
mask), so the default seq is 4096. The optimizer arms should show up as a small
per-step delta (NorMuon/CWD add ~no FLOPs; Smooth-SwiGLU adds a reduction).

WSL2 note: a too-large batch silently spills to host RAM and throughput
collapses rather than OOMing; --mem-fraction caps the allocator so it raises.
"""

from __future__ import annotations

import argparse
import time

import torch

from flower.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from flower.models import build_model
from flower.optim import build_optimizer
from flower.train import autocast_ctx, configure_precision, initialize_lr_schedule


def make_cfg(arm: str, *, d_model: int, layers: int, seq: int, batch: int) -> ExperimentConfig:
    """Build an ExperimentConfig for one benchmark arm.

    All arms share model size / batch / sequence so the comparison is fair; only
    the speedup flags differ.
    """
    model = ModelConfig(
        variant="vanilla_local",
        vocab_size=50257,
        d_model=d_model,
        num_heads=d_model // 64,
        num_layers=layers,
        ffn_dim=d_model * 4,
        max_seq_len=seq,
        local_window=2048,
        norm_type="rmsnorm",
        ffn_activation="swiglu",
        qk_norm=True,
        use_bias=False,
        init_scheme="scaled",
    )
    training = TrainingConfig(
        batch_size=batch,
        steps=10,
        device="cuda",
        precision="bf16",
        optimizer="muon",
        muon_lr=0.02,
    )
    # Apply per-arm flags.
    if arm == "throughput" or arm == "full":
        model.flex_attention = True
        model.fp8_lm_head = True
        model.bf16_cross_entropy = True
    if arm == "full":
        model.smooth_swiglu = True
        model.orthogonal_init = True
        training.norm_update = True
        training.cautious_wd = 0.025
    return ExperimentConfig(model=model, data=DataConfig(sequence_length=seq), training=training)


def bench(cfg: ExperimentConfig, device: torch.device, warmup: int, steps: int, compile_model: bool = False) -> dict:
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    amp_dtype = configure_precision(cfg.training.precision, device)
    model = build_model(cfg.model).to(device)
    optims = build_optimizer(model, cfg.training)
    optims = optims if isinstance(optims, list) else [optims]
    initialize_lr_schedule(optims)

    run = model
    if compile_model:
        # The diagnostics walk uses dir()/getattr over every submodule, which
        # Dynamo cannot trace; disable it (same as train.py).
        if getattr(model, "collect_module_diagnostics", False):
            model.collect_module_diagnostics = False
        run = torch.compile(model, dynamic=False)

    batch = cfg.training.batch_size
    seq = cfg.data.sequence_length
    vocab = cfg.model.vocab_size
    tokens_per_step = batch * seq

    def one_step() -> float:
        for opt in optims:
            opt.zero_grad(set_to_none=True)
        ids = torch.randint(0, vocab, (batch, seq), device=device)
        with autocast_ctx(device, amp_dtype):
            out = run(ids, labels=ids)
            loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        for opt in optims:
            opt.step()
        return float(loss.detach())

    losses = []
    for _ in range(warmup):
        losses.append(one_step())
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        losses.append(one_step())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak = torch.cuda.max_memory_allocated(device) / 1e9
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "tokens_per_sec": steps * tokens_per_step / elapsed,
        "sec_per_step": elapsed / steps,
        "peak_gb": peak,
        "final_loss": losses[-1],
        "finite": all(t == t for t in losses),
        "params_M": params / 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=4096, help="sequence length (longer shows flex win)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--mem-fraction", type=float, default=0.9)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (needed for flex to be fast)")
    parser.add_argument(
        "--arms", type=str, default="legacy,throughput,full",
        help="comma-separated subset of arms to run",
    )
    args = parser.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    label_prefix = "+compile" if args.compile else ""
    print(f"{torch.cuda.get_device_name(0)}" + (" (compiled)" if args.compile else ""))
    print(f"model: d_model={args.d_model}, layers={args.layers}, seq={args.seq}, batch={args.batch}")
    print(f"{'arm':<16} {'tok/s':>10} {'s/step':>9} {'peak GB':>9} {'speedup':>9}  loss")
    print("-" * 66)

    baseline = None
    last = None
    for arm in arms:
        cfg = make_cfg(arm, d_model=args.d_model, layers=args.layers, seq=args.seq, batch=args.batch)
        try:
            r = bench(cfg, device, args.warmup, args.steps, compile_model=args.compile)
        except torch.OutOfMemoryError:
            print(f"{arm+label_prefix:<16} {'OOM':>10}")
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()
        baseline = baseline if baseline is not None else r["tokens_per_sec"]
        flag = "" if r["finite"] else "  NON-FINITE LOSS"
        print(
            f"{arm+label_prefix:<16} {r['tokens_per_sec']:>10,.0f} {r['sec_per_step']:>9.4f} "
            f"{r['peak_gb']:>9.2f} {r['tokens_per_sec'] / baseline:>8.2f}x  {r['final_loss']:.4f}{flag}"
        )
        last = r
    if last is not None:
        print(f"\nparams: ~{last['params_M']:.1f}M")


if __name__ == "__main__":
    main()
