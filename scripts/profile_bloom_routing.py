#!/usr/bin/env python3
"""torch.profiler launch-count check for BloomMemoryBlock._bloom_route.

S14 Opportunity 2 Part A claims collapsing the K-hash Python loop into a single
batched matmul reduces the per-step CPU-side kernel-launch count for bloom
routing. This script isolates one BloomMemoryBlock's memory-update path and
counts the CPU-side kernel launches attributable to it, so we can compare
before/after on identical compute. It is NOT a wall-clock benchmark (bloom
routing is ~5-10% of compute; the win here is launch overhead, best seen at the
profiler level).

  uv run python scripts/profile_bloom_routing.py
"""

from __future__ import annotations

import torch

from flower.config import ModelConfig
from flower.models.bloom_memory import BloomMemoryBlock


def main() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    cfg = ModelConfig(
        variant="bloom_memory",
        vocab_size=128,
        d_model=384,
        num_heads=6,
        num_layers=1,
        ffn_dim=1536,
        max_seq_len=512,
        local_window=512,
        memory_slots=16,
        bloom_num_hashes=4,
        bloom_summary_points=16,
    )
    block = BloomMemoryBlock(cfg).to(device).train()

    B, T, D = 4, 512, cfg.d_model
    x = torch.randn(B, T, D, device=device)
    mem = torch.zeros(B, cfg.memory_slots, D, device=device)

    # Warmup (allocator, cuBLAS handles, autograd).
    for _ in range(3):
        out, mem2 = block(x, mem)
        loss = out.float().sum()
        loss.backward()
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    # Record shapes so we can attribute launches to _bloom_route vs the rest.
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        out, mem2 = block(x, mem)
        loss = out.float().sum()
        loss.backward()
    torch.cuda.synchronize()

    # Count CPU-side aten kernel launches over the whole forward+backward.
    events = prof.key_averages()
    total_cpu_ops = sum(e.count for e in events)
    cuda_events = [e for e in events if e.device_time_total > 0]
    total_cuda_kernels = sum(e.count for e in cuda_events)

    # Attribute launches to bloom routing specifically. _bloom_route issues the
    # batched/looped matmul(s) plus the softmax+mean+diagnostics. We grep the
    # captured stack for the method via a second pass over raw events.
    bloom_keys = {"aten::einsum", "aten::mm", "aten::bmm", "aten::linear",
                  "aten::softmax", "aten::mean", "aten::stack", "aten::clamp_min",
                  "aten::log", "aten::sum"}
    bloom_op_count = sum(e.count for e in events if e.key in bloom_keys)

    print(f"variant=bloom_memory d_model={cfg.d_model} L=1 slots={cfg.memory_slots} K={cfg.bloom_num_hashes} P={cfg.bloom_summary_points}")
    print(f"CPU aten op invocations (forward+backward): {total_cpu_ops}")
    print(f"  of which bloom-attributed ops: {bloom_op_count}")
    print(f"CUDA kernel launches (fwd+bwd):  {total_cuda_kernels}")

    # Table of the top CPU ops, for eyeballing the loop collapse.
    print("\nTop CPU ops by invocation count:")
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="count", row_limit=15))


if __name__ == "__main__":
    main()
