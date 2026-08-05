#!/usr/bin/env python3
"""Benchmark Liger fused RMSNorm + SwiGLU vs the current eager/compiled path.

Question: is adopting LigerRMSNorm + LigerSwiGLUMLP (drop-in replacements) worth
it for flower? The doc (§5b) ranks them P1 with modest expected gains, and §5a
notes the optimizer step was launch-bound not compute-bound. This measures the
actual per-component and end-to-end-step delta at the production long-context
shape, under torch.compile (the production setting), so the decision is data.

Both paths compared:
  current : flower's RMSNorm + 3x nn.Linear SwiGLU (flower/models/base.py)
  liger   : LigerRMSNorm + LigerSwiGLUMLP (fused Triton kernels)

Each is timed in isolation (just the norm; just the FFN) AND as a full training
step (fwd+bwd+opt) to see whether the kernel speedup survives the surrounding
matmuls/attention dominating the step.

USAGE
  PYTHONPATH=. uv run python scripts/bench_liger_kernels.py
  PYTHONPATH=. uv run python scripts/bench_liger_kernels.py --seq 8192 --layers 14 --d 768
"""
from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from flower.models.base import RMSNorm, swiglu_hidden_dim
from flower.config import ModelConfig


# ---------------------------------------------------------------------------
# Stubs so LigerSwiGLUMLP (HF-config-shaped) can be built from raw dims.
# ---------------------------------------------------------------------------

class _SwiGLUConfig:
    """Minimal config object satisfying LigerSwiGLUMLP.__init__."""
    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = "silu"


def build_current_ffn(d_model, hidden, bias=False, dropout=0.0):
    """flower's current SwiGLU: 3 nn.Linear + F.silu + elementwise mul."""
    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(d_model, hidden, bias=bias)
            self.up = nn.Linear(d_model, hidden, bias=bias)
            self.down = nn.Linear(hidden, d_model, bias=bias)
        def forward(self, x):
            return self.down(F.silu(self.gate(x)) * self.up(x))
    return FFN()


def build_liger_ffn(d_model, hidden):
    from liger_kernel.transformers import LigerSwiGLUMLP
    return LigerSwiGLUMLP(_SwiGLUConfig(d_model, hidden))


def build_current_rmsnorm(d_model):
    return RMSNorm(d_model)


def build_liger_rmsnorm(d_model):
    from liger_kernel.transformers import LigerRMSNorm
    return LigerRMSNorm(hidden_size=d_model, eps=1e-6)


def _time(fn, warmup, iters, device):
    """Time `fn` (fwd+bwd) over `iters` after `warmup` warmups. Returns ms/iter."""
    for _ in range(warmup):
        out = fn()
        out.sum().backward()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        out.sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def bench_component(name, make_fn, x_shape, device, dtype, compile_it, warmup=10, iters=30):
    """Compare current vs liger for a single component (norm or ffn)."""
    torch.manual_seed(0)
    x = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)

    results = {}
    for label, build in make_fn.items():
        torch.manual_seed(0)
        mod = build().to(device).to(dtype)
        # match the grad-requiring input
        xx = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)
        fn = lambda m=mod, inp=xx: m(inp)
        if compile_it:
            fn = torch.compile(fn, mode="default", dynamic=False)
        try:
            ms = _time(fn, warmup, iters, device)
            results[label] = ms
            del mod
        except Exception as e:
            results[label] = None
            results[f"{label}_err"] = str(e)[:80]
        gc.collect(); torch.cuda.empty_cache()
    return results


def bench_full_step(d_model, num_layers, seq, batch, device, dtype, ffn_kind, norm_kind, compile_it, warmup=10, iters=20):
    """A minimal transformer stack (embed + N x [norm, ffn] + head) fwd+bwd+sgd step."""
    torch.manual_seed(0)
    vocab = 4096
    hidden = swiglu_hidden_dim(ModelConfig(ffn_activation="swiglu", ffn_param_match=True, ffn_dim=d_model*4), d_model*4)

    embed = nn.Embedding(vocab, d_model).to(device)
    norm_fn = build_current_rmsnorm if norm_kind == "current" else build_liger_rmsnorm
    ffn_fn = (lambda: build_current_ffn(d_model, hidden)) if ffn_kind == "current" else (lambda: build_liger_ffn(d_model, hidden))
    norms = nn.ModuleList([norm_fn(d_model) for _ in range(num_layers)]).to(device).to(dtype)
    ffns = nn.ModuleList([ffn_fn() for _ in range(num_layers)]).to(device).to(dtype)
    head = nn.Linear(d_model, vocab, bias=False).to(device).to(dtype)
    params = list(embed.parameters()) + list(norms.parameters()) + list(ffns.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=1e-3)

    def step():
        ids = torch.randint(0, vocab, (batch, seq), device=device)
        x = embed(ids).to(dtype)  # cast embedding output to bf16 to match the stack
        for nm, fm in zip(norms, ffns):
            x = x + fm(nm(x))
        logits = head(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    if compile_it:
        step = torch.compile(step, mode="default", dynamic=False)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = []
    for _ in range(iters):
        losses.append(float(step()))
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    peak = torch.cuda.max_memory_allocated() / 1e9
    tokens = batch * seq
    return elapsed * 1000, tokens / elapsed, peak, losses[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--layers", type=int, default=14)
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--no-compile", action="store_true", help="also bench eager (no compile)")
    args = p.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    d, L, seq, batch = args.d, args.layers, args.seq, args.batch
    hidden = swiglu_hidden_dim(ModelConfig(ffn_activation="swiglu", ffn_param_match=True, ffn_dim=d*4), d*4)
    print(f"shape: d={d} hidden={hidden} L={L} seq={seq} batch={batch} bf16")
    print(f"flex/attention NOT included here — isolates norm+FFN kernel cost\n")

    for compile_it in ([True, False] if args.no_compile else [True]):
        tag = "compiled" if compile_it else "eager"
        print(f"=== component benchmark ({tag}) ===")
        # RMSNorm: (batch, seq, d)
        r = bench_component("RMSNorm",
            {"current": lambda: build_current_rmsnorm(d), "liger": lambda: build_liger_rmsnorm(d)},
            (batch, seq, d), device, dtype, compile_it)
        print(f"  RMSNorm  current={r.get('current')}ms  liger={r.get('liger')}ms"
              + (f"  ({r['liger_err']})" if r.get('liger_err') else
                 (f"  speedup={r['current']/r['liger']:.2f}x" if r.get('current') and r.get('liger') else "")))
        # FFN: (batch, seq, d) -> (batch, seq, d)
        r = bench_component("SwiGLU",
            {"current": lambda: build_current_ffn(d, hidden), "liger": lambda: build_liger_ffn(d, hidden)},
            (batch, seq, d), device, dtype, compile_it)
        print(f"  SwiGLU   current={r.get('current')}ms  liger={r.get('liger')}ms"
              + (f"  ({r['liger_err']})" if r.get('liger_err') else
                 (f"  speedup={r['current']/r['liger']:.2f}x" if r.get('current') and r.get('liger') else "")))
        print()

    print("=== full training step (fwd+bwd+opt, compiled) — current vs liger norm+FFN ===")
    print(f"    {L} layers, no attention (isolates the norm+FFN contribution)\n")
    rows = []
    for ffn in ["current", "liger"]:
        for norm in ["current", "liger"]:
            try:
                ms, tps, peak, loss = bench_full_step(d, L, seq, batch, device, dtype, ffn, norm, compile_it=True)
                rows.append((ffn, norm, ms, tps, peak, loss))
                print(f"  ffn={ffn:7s} norm={norm:7s}: {ms:.1f} ms/step  {tps:.0f} tok/s  peak={peak:.2f}GB  loss={loss:.3f}")
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ffn={ffn} norm={norm}: ERROR {str(e)[:80]}")
                gc.collect(); torch.cuda.empty_cache()

    if len(rows) >= 2:
        base = next(r for r in rows if r[0]=="current" and r[1]=="current")
        all_liger = next((r for r in rows if r[0]=="liger" and r[1]=="liger"), None)
        if all_liger:
            sp = all_liger[3] / base[3]
            print(f"\n  all-liger vs all-current: {sp:.3f}x tok/s  ({(sp-1)*100:+.1f}%)  "
                  f"memory {all_liger[4]:.2f} vs {base[4]:.2f} GB")


if __name__ == "__main__":
    main()
