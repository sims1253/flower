"""Benchmark selective vs full vs none activation checkpointing.

Each invocation measures ONE mode and ONE shape in a fresh process (the GPU is
flaky under multi-build concurrency, so we do not loop). Prints a single
machine-readable line:

    RESULT mode=<none|full|selective> shape=<tag> loss=<f> peak_gb=<f> tok_s=<f> ms_step=<f>

Run this script three times (once per mode) and collect the RESULT lines.

Usage:
    uv run python scripts/bench_selective_checkpoint.py <mode> <shape>

    mode : none | full | selective
    shape: 8k   (100M, d768/L14, seq8192, b4)
            32k  (100M, d768/L14, seq32768, b1)
"""
from __future__ import annotations

import gc
import statistics
import sys
import time

import torch

from flower.config import ModelConfig
from flower.models import build_model
from flower.models.base import prebuild_attention_masks

_MODE_TO_CFG = {"none": False, "full": True, "selective": "selective"}

_SHAPES = {
    "8k": dict(seq=8192, batch=4),
    "32k": dict(seq=32768, batch=1),
}


def make_cfg(activation_checkpoint, seq):
    return ModelConfig(
        variant="vanilla_local",
        vocab_size=4096,
        d_model=768,
        num_heads=12,
        num_layers=14,
        ffn_dim=3072,
        max_seq_len=seq,
        local_window=2048,
        dropout=0.0,
        norm_type="rmsnorm",
        ffn_activation="swiglu",
        ffn_param_match=True,
        qk_norm=True,
        use_bias=False,
        init_scheme="scaled",
        init_std=0.02,
        flex_attention=True,
        activation_checkpoint=activation_checkpoint,
    )


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "none"
    shape = sys.argv[2] if len(sys.argv) > 2 else "8k"
    if mode not in _MODE_TO_CFG:
        raise SystemExit(f"mode must be one of {list(_MODE_TO_CFG)}, got {mode!r}")
    if shape not in _SHAPES:
        raise SystemExit(f"shape must be one of {list(_SHAPES)}, got {shape!r}")

    seq = _SHAPES[shape]["seq"]
    batch = _SHAPES[shape]["batch"]
    dev = torch.device("cuda")

    torch.manual_seed(0)
    cfg = make_cfg(_MODE_TO_CFG[mode], seq)
    model = build_model(cfg).to(dev).to(torch.bfloat16)
    prebuild_attention_masks(model, seq, dev)
    model.train()

    ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=dev)
    compiled = torch.compile(model, mode="default", dynamic=False)

    # Warmup: 2 steps so compile tracing / cudagraph warmup is excluded from
    # the timed region. Use fresh inputs each step (same distribution).
    torch.manual_seed(100)
    for _ in range(2):
        warm_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = compiled(warm_ids, labels=warm_ids)["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)

    # Measure memory on a representative fwd+bwd.
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(200)
    meas_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = compiled(meas_ids, labels=meas_ids)["loss"]
    loss_val = float(loss.detach())
    loss.backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    model.zero_grad(set_to_none=True)

    # Throughput: time N steps, report median ms/step and tok/s.
    n_steps = 5
    timings_ms = []
    torch.manual_seed(300)
    for _ in range(n_steps):
        step_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = compiled(step_ids, labels=step_ids)["loss"]
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)
        model.zero_grad(set_to_none=True)

    ms_step = statistics.median(timings_ms)
    tok_s = (batch * seq) / (ms_step / 1000.0)

    print(
        f"RESULT mode={mode} shape={shape} loss={loss_val:.6f} "
        f"peak_gb={peak_gb:.3f} tok_s={tok_s:.0f} ms_step={ms_step:.1f}",
        flush=True,
    )

    del model, compiled, ids, meas_ids
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
