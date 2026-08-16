"""Specialized bf16 sliding-window causal attention (Triton), forward + backward.

WHY THIS EXISTS
  After FP8 takes the linear layers (1.275x, docs/profiling/speedup_results.md),
  attention is the largest remaining block of the 450M step: flex_attention's
  backward is 20.8% of CUDA time and its forward 6.5%. FlexAttention's own
  block-size and kernel-option space was swept and is already optimal for this
  shape (scripts/bench_flex_config.py). So the only lever left is replacing it.

THE MEASUREMENT THAT MOTIVATED THIS (B1 H20 T4096 D64 window 2048, RTX 5090)

      flex_attention forward (compiled)    0.565 ms   1.00x    —
      FP8 specialized forward              0.431 ms   1.31x    18x bf16 error
      bf16 specialized forward             0.380 ms   1.49x    0.0009 error

  The FP8 kernel came first, on the theory that FP8 was the win. It is not:
  **the speedup is specialization, not precision.** bf16 is both FASTER than FP8
  here (FP8's amax reductions and casts cost more than its tensor-core advantage
  at head_dim 64 tiles) and numerically free. See
  `flower/kernels/fp8_swa_attention.py` for that kernel and the numerics table
  that ruled FP8 out of attention entirely.

  Where the 1.49x comes from: flex is general. It evaluates a user `mask_mod`
  through generated code and supports arbitrary block masks and score_mods.
  A kernel that hardcodes "causal + fixed window" skips all of that bookkeeping
  and computes a tighter kv range per query block.

SCOPE / LIMITS
  - Causal + fixed sliding window ONLY. No score_mod. The RBF-bias path
    (`kernel_bias == "rbf"`) and any full-context layer must keep using flex —
    this is not a drop-in for every attention configuration in the codebase.
  - head_dim must be 32/64/128 (tl.dot tiling).
  - bf16/fp16 in, same dtype out; softmax and accumulation in fp32.

OUTCOME: NET LOSS. DO NOT WIRE IN.
  Forward and backward are implemented and verified correct against flex (values
  and all three gradients agree to bf16 rounding, ~0.0005 relative). But the
  complete, differentiable kernel is SLOWER than flex end to end:

      flex          fwd 0.580 ms   fwd+bwd 2.047 ms
      this kernel   fwd 0.614 ms   fwd+bwd 2.196 ms
                    0.94x          0.93x

  The 1.49x forward that motivated this does not survive being made usable:

    1. That measurement was a forward with NO logsumexp output. Any backward
       needs L = m + log(l) saved, and adding the L computation and store takes
       the raw forward from 0.380 ms to ~0.71 ms. A forward that cannot
       back-propagate is not a training speedup.
    2. The backward — which is where the prize is, 20.8% of CUDA time vs the
       forward's 6.5% — is slower than flex's. flex's
       `flex_attention_backward_split_transpose` is a split-reduction template
       that inductor autotunes; a straightforward two-kernel dK/dV + dQ
       implementation does not match it.

  Tuning mattered enormously and in non-obvious ways, so for anyone revisiting:
  an untuned launch geometry was 9x SLOWER than flex; a single config shared
  across the three kernels gave 0.90x; per-kernel configs gave 0.93x. The
  configs below are the per-kernel optima. The remaining 7% would need warp
  specialisation / persistent-kernel work, not parameter search.

  Kept in-tree as a verified reference implementation and as the evidence that
  closes this direction — together with `fp8_swa_attention.py`, which closes the
  FP8 variant on numerics. Between them: flex is not beatable here by either
  precision or specialisation.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _swa_fwd(
        Q, K, V, Out, L,
        sm_scale,
        sqb, sqh, sqm, sqd, skb, skh, skn, skd,
        svb, svh, svn, svd, sob, soh, som, sod,
        slb, slh, slm,
        H: tl.constexpr, N_CTX: tl.constexpr, WINDOW: tl.constexpr,
        HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Flash-attention v2 forward. Also writes L = m + log(l) for the backward."""
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // H
        off_h = off_bh % H

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        q = tl.load(
            Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
            mask=offs_m[:, None] < N_CTX, other=0.0,
        )

        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        # Query i attends to keys in (i - WINDOW, i]. Only kv blocks overlapping
        # that band are visited — this is the whole point of the window.
        lo = tl.maximum(0, start_m * BLOCK_M - WINDOW + 1)
        lo = (lo // BLOCK_N) * BLOCK_N
        hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)

        for start_n in range(lo, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                K + off_b * skb + off_h * skh + offs_n[:, None] * skn + offs_d[None, :] * skd,
                mask=offs_n[:, None] < N_CTX, other=0.0,
            )
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
            valid = (
                (offs_m[:, None] >= offs_n[None, :])
                & ((offs_m[:, None] - offs_n[None, :]) < WINDOW)
                & (offs_n[None, :] < N_CTX)
            )
            qk = tl.where(valid, qk, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            # An all-masked row leaves m_new = -inf; clamping keeps exp() finite
            # and makes the row contribute nothing.
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.where(valid, tl.exp(qk - m_safe[:, None]), 0.0)
            alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_safe))

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v = tl.load(
                V + off_b * svb + off_h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd,
                mask=offs_n[:, None] < N_CTX, other=0.0,
            )
            acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_safe[:, None]

        tl.store(
            Out + off_b * sob + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sod,
            acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX,
        )
        # Logsumexp, in the combined form the backward needs: P = exp(S - L).
        m_store = tl.where(m_i == -float("inf"), 0.0, m_i)
        tl.store(
            L + off_b * slb + off_h * slh + offs_m * slm,
            m_store + tl.log(l_safe), mask=offs_m < N_CTX,
        )

    @triton.jit
    def _swa_bwd_preprocess(
        Out, DO, Delta,
        sob, soh, som, sod, sdb, sdh, sdm, sdd, sdeb, sdeh, sdem,
        H: tl.constexpr, N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        """Delta = rowsum(dO * O) — the term that turns dP into dS."""
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // H
        off_h = off_bh % H
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        mask = offs_m[:, None] < N_CTX
        o = tl.load(Out + off_b * sob + off_h * soh + offs_m[:, None] * som + offs_d[None, :] * sod,
                    mask=mask, other=0.0).to(tl.float32)
        do = tl.load(DO + off_b * sdb + off_h * sdh + offs_m[:, None] * sdm + offs_d[None, :] * sdd,
                     mask=mask, other=0.0).to(tl.float32)
        tl.store(Delta + off_b * sdeb + off_h * sdeh + offs_m * sdem,
                 tl.sum(o * do, 1), mask=offs_m < N_CTX)

    @triton.jit
    def _swa_bwd_dkdv(
        Q, K, V, DO, DK, DV, L, Delta, sm_scale,
        sqb, sqh, sqm, sqd, skb, skh, skn, skd, svb, svh, svn, svd,
        sdob, sdoh, sdom, sdod, slb, slh, slm,
        H: tl.constexpr, N_CTX: tl.constexpr, WINDOW: tl.constexpr,
        HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """One program per kv block; loops over the query blocks that see it.

        Key j is attended to by queries i in [j, j + WINDOW). Iterating that band
        (rather than all queries) is what keeps the backward O(T*W).
        """
        start_n = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // H
        off_h = off_bh % H

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)

        k = tl.load(K + off_b * skb + off_h * skh + offs_n[:, None] * skn + offs_d[None, :] * skd,
                    mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(V + off_b * svb + off_h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd,
                    mask=offs_n[:, None] < N_CTX, other=0.0)

        dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
        hi = tl.minimum(N_CTX, (start_n + 1) * BLOCK_N - 1 + WINDOW)

        for start_m in range(lo, hi, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            m_mask = offs_m < N_CTX
            q = tl.load(Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                        mask=m_mask[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
            valid = (
                (offs_m[:, None] >= offs_n[None, :])
                & ((offs_m[:, None] - offs_n[None, :]) < WINDOW)
                & m_mask[:, None] & (offs_n[None, :] < N_CTX)
            )
            lse = tl.load(L + off_b * slb + off_h * slh + offs_m * slm, mask=m_mask, other=0.0)
            p = tl.where(valid, tl.exp(qk - lse[:, None]), 0.0)

            do = tl.load(DO + off_b * sdob + off_h * sdoh + offs_m[:, None] * sdom + offs_d[None, :] * sdod,
                         mask=m_mask[:, None], other=0.0)
            dv += tl.dot(tl.trans(p).to(do.dtype), do, out_dtype=tl.float32)

            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
            delta = tl.load(Delta + off_b * slb + off_h * slh + offs_m * slm, mask=m_mask, other=0.0)
            ds = tl.where(valid, p * (dp - delta[:, None]) * sm_scale, 0.0)
            dk += tl.dot(tl.trans(ds).to(q.dtype), q, out_dtype=tl.float32)

        tl.store(DK + off_b * skb + off_h * skh + offs_n[:, None] * skn + offs_d[None, :] * skd,
                 dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < N_CTX)
        tl.store(DV + off_b * svb + off_h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd,
                 dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < N_CTX)

    @triton.jit
    def _swa_bwd_dq(
        Q, K, V, DO, DQ, L, Delta, sm_scale,
        sqb, sqh, sqm, sqd, skb, skh, skn, skd, svb, svh, svn, svd,
        sdob, sdoh, sdom, sdod, slb, slh, slm,
        H: tl.constexpr, N_CTX: tl.constexpr, WINDOW: tl.constexpr,
        HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """One program per query block; loops over the kv blocks it attends to."""
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // H
        off_h = off_bh % H

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        m_mask = offs_m < N_CTX

        q = tl.load(Q + off_b * sqb + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                    mask=m_mask[:, None], other=0.0)
        do = tl.load(DO + off_b * sdob + off_h * sdoh + offs_m[:, None] * sdom + offs_d[None, :] * sdod,
                     mask=m_mask[:, None], other=0.0)
        lse = tl.load(L + off_b * slb + off_h * slh + offs_m * slm, mask=m_mask, other=0.0)
        delta = tl.load(Delta + off_b * slb + off_h * slh + offs_m * slm, mask=m_mask, other=0.0)

        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        lo = tl.maximum(0, start_m * BLOCK_M - WINDOW + 1)
        lo = (lo // BLOCK_N) * BLOCK_N
        hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)

        for start_n in range(lo, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(K + off_b * skb + off_h * skh + offs_n[:, None] * skn + offs_d[None, :] * skd,
                        mask=offs_n[:, None] < N_CTX, other=0.0)
            v = tl.load(V + off_b * svb + off_h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd,
                        mask=offs_n[:, None] < N_CTX, other=0.0)
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
            valid = (
                (offs_m[:, None] >= offs_n[None, :])
                & ((offs_m[:, None] - offs_n[None, :]) < WINDOW)
                & m_mask[:, None] & (offs_n[None, :] < N_CTX)
            )
            p = tl.where(valid, tl.exp(qk - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
            ds = tl.where(valid, p * (dp - delta[:, None]) * sm_scale, 0.0)
            dq += tl.dot(ds.to(k.dtype), k, out_dtype=tl.float32)

        tl.store(DQ + off_b * sqb + off_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
                 dq.to(DQ.dtype.element_ty), mask=m_mask[:, None])


# Launch geometries, tuned INDEPENDENTLY per kernel at the 450M shape
# (B*H=20, T=4096, head_dim 64, window 2048) on an RTX 5090.
#
# Tuning them jointly is a measurable mistake: the forward prefers 128x32 with
# 8 warps while both backward kernels prefer 64x32 with 4 warps, and forcing one
# config on all three produced 0.90x vs flex (a net LOSS) where per-kernel
# configs give a win. Each tuple is (BLOCK_M, BLOCK_N, num_warps, num_stages).
_FWD_CFG = (128, 32, 8, 3)
_DKDV_CFG = (64, 32, 4, 3)
_DQ_CFG = (64, 32, 4, 3)


class _SlidingWindowAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window, fwd_cfg, dkdv_cfg, dq_cfg):
        B, H, T, D = q.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = torch.empty_like(q)
        lse = torch.empty((B, H, T), device=q.device, dtype=torch.float32)

        bm, bn, w, s = fwd_cfg
        _swa_fwd[(triton.cdiv(T, bm), B * H)](
            q, k, v, out, lse, 1.0 / (D ** 0.5),
            *q.stride(), *k.stride(), *v.stride(), *out.stride(), *lse.stride(),
            H=H, N_CTX=T, WINDOW=window, HEAD_DIM=D,
            BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=s,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.window = window
        ctx.dkdv_cfg, ctx.dq_cfg = dkdv_cfg, dq_cfg
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        B, H, T, D = q.shape
        do = do.contiguous()
        window = ctx.window
        sm_scale = 1.0 / (D ** 0.5)

        delta = torch.empty_like(lse)
        _swa_bwd_preprocess[(triton.cdiv(T, 128), B * H)](
            out, do, delta, *out.stride(), *do.stride(), *delta.stride(),
            H=H, N_CTX=T, HEAD_DIM=D, BLOCK_M=128,
        )

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        bm, bn, w, s = ctx.dkdv_cfg
        _swa_bwd_dkdv[(triton.cdiv(T, bn), B * H)](
            q, k, v, do, dk, dv, lse, delta, sm_scale,
            *q.stride(), *k.stride(), *v.stride(), *do.stride(), *lse.stride(),
            H=H, N_CTX=T, WINDOW=window, HEAD_DIM=D,
            BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=s,
        )
        bm, bn, w, s = ctx.dq_cfg
        _swa_bwd_dq[(triton.cdiv(T, bm), B * H)](
            q, k, v, do, dq, lse, delta, sm_scale,
            *q.stride(), *k.stride(), *v.stride(), *do.stride(), *lse.stride(),
            H=H, N_CTX=T, WINDOW=window, HEAD_DIM=D,
            BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=s,
        )
        return dq, dk, dv, None, None, None, None


def swa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
    *, fwd_cfg=_FWD_CFG, dkdv_cfg=_DKDV_CFG, dq_cfg=_DQ_CFG,
) -> torch.Tensor:
    """Causal sliding-window attention. q/k/v: (B, H, T, D). Differentiable.

    Defaults are tuned per-kernel for the 450M shape. They are not arbitrary: an
    untuned launch geometry measured 9x SLOWER than flex, and a single shared
    config across the three kernels measured 0.90x (a loss).
    """
    if not _HAS_TRITON:
        raise RuntimeError("swa_attention requires Triton")
    if q.dim() != 4:
        raise ValueError(f"expected (B, H, T, D), got {tuple(q.shape)}")
    if q.shape[-1] not in (32, 64, 128):
        raise ValueError(f"head_dim must be 32/64/128, got {q.shape[-1]}")
    if window <= 0:
        raise ValueError("window must be positive; use flex for full-context attention")
    return _SlidingWindowAttention.apply(q, k, v, window, fwd_cfg, dkdv_cfg, dq_cfg)
