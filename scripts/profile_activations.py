#!/usr/bin/env python3
"""Profile activation memory breakdown for a single transformer block.

Goal: figure out WHICH activations dominate memory at long context, so we can
design a smarter checkpointing policy than "drop everything" (full) or "drop by
byte-size" (selective). The hypothesis to test: at seq=32K, the attention
Q/K/V/O tensors (4 x B x T x d) dominate storage but the FFN intermediate
(B x T x ffn_dim) is also large; meanwhile the MATMULS are the recompute cost.
A good policy drops the storage-hogs but avoids recomputing the most expensive
matmuls. This script measures both dimensions.

Reports, per block, for the production shape:
  - every saved activation tensor (name, shape, bytes) — the STORAGE profile
  - approximate FLOPs of each matmul in attn + FFN — the RECOMPUTE profile
  - the storage/flops ratio (what you save in memory per flop of recompute)

USAGE
  PYTHONPATH=. uv run python scripts/profile_activations.py --seq 32768 --d 768
  PYTHONPATH=. uv run python scripts/profile_activations.py --seq 8192 --d 768 --batch 4
"""
from __future__ import annotations
import argparse, torch
import torch.nn.functional as F

def sz(t: torch.Tensor) -> float:
    return t.nelement() * t.element_size() / 1e6  # MB

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=32768)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--ffn", type=int, default=2048)  # swiglu hidden (2/3*3072 rounded)
    args = p.parse_args()

    B, T, D, H, Fh = args.batch, args.seq, args.d, args.heads, args.ffn
    hd = D // H
    print(f"shape: B={B} T={T} D={D} H={H} head_dim={hd} ffn_hidden={Fh}  (bf16, 2 bytes/elem)")
    print(f"per-token-positions: {T}  total tokens/batch: {B*T}")
    print()

    # ---- STORAGE: what activations a backward pass needs to retain (per block) ----
    print("=== STORAGE: activations retained for backward (per block, bf16) ===")
    acts = []
    # attention path
    acts.append(("attn.qkv_in (x)", (B, T, D)))           # input to qkv proj
    acts.append(("attn.q", (B, H, T, hd))); acts.append(("attn.k", (B, H, T, hd))); acts.append(("attn.v", (B, H, T, hd)))
    acts.append(("attn.softmax_stats (lse)", (B, H, T, 1)))  # flash/flex saves logsumexp, fp32
    acts.append(("attn.out_pre (B,H,T,hd)", (B, H, T, hd)))
    acts.append(("attn.out_proj_in", (B, T, D)))
    # ffn path
    acts.append(("ffn.gate_in (x)", (B, T, D)))
    acts.append(("ffn.up", (B, T, Fh)))                    # the big one
    acts.append(("ffn.gate", (B, T, Fh)))
    acts.append(("ffn.act (silu(gate))", (B, T, Fh)))
    acts.append(("ffn.fused (act*up)", (B, T, Fh)))        # checkpointed by selective
    acts.append(("ffn.down_in", (B, T, Fh)))
    acts.append(("ffn.down_out", (B, T, D)))
    # residual/norm (small)
    acts.append(("residual x (ln1 in)", (B, T, D)))

    total = 0.0
    for name, shape in acts:
        n = 1
        for s in shape: n *= s
        mb = n * 2 / 1e6  # bf16
        # lse is fp32
        if "lse" in name or "stats" in name: mb = n * 4 / 1e6
        total += mb
        flag = " <-- BIG" if mb > total * 0.08 else ""
        print(f"  {name:32s} {str(shape):22s} {mb:8.1f} MB{flag}")
    print(f"  {'TOTAL per block':32s} {'':22s} {total:8.1f} MB")
    print()

    # ---- RECOMPUTE: FLOPs of each matmul (2*in*out per token, fwd) ----
    print("=== RECOMPUTE: matmul FLOPs if this op is recomputed (per block, fwd only) ===")
    # matmul flops = 2 * M * N * K for (M,K)@(K,N); per-token M=B*T
    def flops(din, dout): return 2 * B * T * din * dout
    ops = [
        ("attn.qkv_proj", D, 3 * D),
        ("attn.out_proj", D, D),
        ("attn.scores (Q@K^T per head)", None, None),  # special: 2*B*H*T*T*hd
        ("attn.attn@V per head", None, None),          # 2*B*H*T*T*hd
        ("ffn.gate_proj", D, Fh),
        ("ffn.up_proj", D, Fh),
        ("ffn.down_proj", Fh, D),
    ]
    total_flops = 0
    attn_score_flops = 2 * B * H * T * T * hd
    for name, din, dout in ops:
        if din is None:
            f = attn_score_flops
            tag = f"{f/1e9:.2f} GFLOP"
        else:
            f = flops(din, dout)
            tag = f"{f/1e9:.2f} GFLOP ({din}->{dout})"
        total_flops += f
        print(f"  {name:28s} {tag}")
    # backward is ~2x forward (recompute + grad), so checkpoint adds ~forward flops
    print(f"  {'TOTAL fwd FLOPs per block':28s} {total_flops/1e9:.2f} GFLOP")
    print(f"  (full checkpoint adds ~{total_flops/1e9:.2f} GFLOP/block recompute = ~{100*total_flops/(2*total_flops):.0f}% of fwd+bwd)")
    print()

    # ---- The ratio: what each saved tensor costs to recompute ----
    print("=== STORAGE / RECOMPUTE ratio (MB saved per GFLOP recompute) ===")
    print("(high = efficient to checkpoint: lots of memory freed, little recompute)")
    # FFN: saving 'fused' (B,T,Fh) lets you skip storing up/gate/down_in, but recompute = up+gate+down matmuls")
    ffn_storage = sz(torch.empty(B, T, Fh)) * 3 / 1e6 * 0  # placeholder
    ffn_save_mb = (B * T * Fh * 2 / 1e6) * 3  # up + act + down_in roughly
    ffn_recompute_gf = (flops(D, Fh) + flops(D, Fh) + flops(Fh, D)) / 1e9
    print(f"  FFN block (drop intermediates): save ~{ffn_save_mb:.1f} MB, recompute {ffn_recompute_gf:.2f} GFLOP  ratio={ffn_save_mb/ffn_recompute_gf:.1f} MB/GFLOP")
    # Attention: Q/K/V/O storage but the score matmuls are O(T^2) — VERY expensive to recompute at long T")
    attn_save_mb = (B * T * D * 2 / 1e6) * 3  # qkv + out roughly
    attn_recompute_gf = (flops(D, 3*D) + flops(D, D) + 2*attn_score_flops) / 1e9
    print(f"  Attn block (drop q/k/v/out):     save ~{attn_save_mb:.1f} MB, recompute {attn_recompute_gf:.2f} GFLOP  ratio={attn_save_mb/attn_recompute_gf:.1f} MB/GFLOP")
    print()
    print(f"  ==> At long context (T={T}), attention recompute is {attn_recompute_gf/ffn_recompute_gf:.1f}x the FFN recompute")
    print(f"      but saves similar memory. So a policy that AVOIDS recomputing attention")
    print(f"      (keeps q/k/v/o, only checkpoints the FFN) should beat full checkpointing.")


if __name__ == "__main__":
    main()
