"""FP8 sliding-window causal attention (Triton), forward pass.

WHY THIS EXISTS
  After FP8 takes the linear layers (1.275x, see docs/profiling/speedup_results.md),
  attention is the largest remaining block of the 450M step — `flex_attention`
  forward + backward together are ~27% of CUDA time. FlexAttention has no FP8
  path, and its block-size / kernel-option space was swept and found already
  optimal (scripts/bench_flex_config.py: best alternative 1.028x, inside noise).
  So the only remaining lever on attention is a hand-written FP8 kernel.

  Feasibility was measured before writing any of this
  (scripts/bench_fp8_attention_feasibility.py): at attention's actual inner tile
  shapes, FP8 beats bf16 by 1.88x on QK^T and 1.64x on PV. Triton 3.7.1's
  `tl.dot` with `float8_e4m3fn` was verified exact on sm_120 first.

WHY FORWARD ONLY (for now)
  The backward is 20.8% of CUDA time and the forward only 6.5%, so the backward
  is where the prize is — but gradients have a far wider dynamic range than
  activations, and FP8 attention backward is where published implementations
  (e.g. FlashAttention-3) stop as well. Writing the forward first makes the
  question cheap to answer: if the forward does not comfortably beat flex's
  forward, the backward certainly will not pay for its much larger risk, and the
  whole direction is closed for the cost of one kernel.

WHY FP8 IS PLAUSIBLE FOR *THIS* MODEL
  `qk_norm: true` (S12.1) applies RMSNorm to Q and K per head, so their entries
  are bounded rather than heavy-tailed — which is exactly the condition under
  which per-tensor e4m3 quantization is safe. A model without QK-norm would be a
  much worse candidate.

NUMERICS
  - Q, K, V are quantized to e4m3 with per-tensor scales computed from amax.
  - Scores accumulate in fp32; the online-softmax running max/sum stay fp32.
  - P (softmax probabilities, in [0, 1]) is rescaled by 448.0 (the e4m3 max)
    before quantizing, so the probability range uses the full mantissa rather
    than the bottom of it. The compensating 1/448 folds into the V scale.
  - The output accumulator is fp32 and is written back as bf16.

MEASURED (B1 H20 T4096 D64 window 2048, RTX 5090)
  SPEED — the kernel wins:
      flex_attention forward (compiled, tuned)   0.588 ms
      this kernel, default BLOCK_M/N 128/64      5.555 ms   0.11x  (!)
      this kernel, tuned  64/32 warps4 stages3   0.431 ms   1.36x
    The untuned draft was 9x SLOWER than flex; tuning the launch geometry moved
    it to 1.36x faster. Any conclusion drawn from an untuned Triton kernel is
    worthless — hence the defaults above are the tuned ones.

  NUMERICS — the kernel loses, and this is the blocker:
    Mean relative error of the attention output vs an fp32 reference, measured
    as pure input-quantization error (fp32 ground-truth inputs):

      bf16 Q,K,V (what the model does today)   0.00277    1.0x
      fp8 V only          (Q,K,P bf16)         0.02611    9.4x
      fp8 P only          (Q,K,V bf16)         0.02612    9.4x
      fp8 Q,K only        (V,P bf16)           0.03534   12.8x
      fp8 Q,K,V,P         (this kernel)        0.05110   18.4x
      fp8 Q,K,V,P         per-head scales      0.05089   18.4x

    Two things follow. First, there is no partial-FP8 configuration that gets
    near bf16 — putting even ONE tensor in e4m3 costs 9x, and the cheapest
    variants also accelerate only one of the two matmuls. Second, per-head
    scaling does not help *at all* (18.4x either way), which identifies the
    error as MANTISSA-limited, not scale-limited: e4m3 has 3 mantissa bits and
    attention reduces over only head_dim=64, so quantization noise does not
    average down the way it does in the linear layers (K = 1280-3392). No
    scaling scheme fixes this.

  For context on why FP8 was fine for the linear layers but not here: FP8 linear
  cost only +0.001 val_bpb, on reductions 20-50x longer than head_dim.

STATUS: forward implemented, tuned, and correctness-tested against an fp32
reference. **NOT wired into any model and should not be** on this evidence —
`flower/models/base.py` still calls flex. The open question is whether ~5%
attention-output error actually degrades trained quality; answering it needs a
backward pass (this is forward-only), so the cheap way to find out first is the
straight-through fake-quant screen rather than more kernel code.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_TRITON = False


E4M3_MAX = 448.0


if _HAS_TRITON:

    @triton.jit
    def _fp8_swa_fwd_kernel(
        Q, K, V, Out,
        qk_scale,          # fp32 scalar: q_scale * k_scale * softmax_scale
        pv_scale,          # fp32 scalar: v_scale / 448.0
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        H: tl.constexpr,
        N_CTX: tl.constexpr,
        WINDOW: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One program per (query block, batch*head). Flash-attention v2 style."""
        start_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        off_b = off_bh // H
        off_h = off_bh % H

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        # Sliding window: query i attends to keys in (i - WINDOW, i]. Only the
        # kv blocks overlapping that band are visited — this is where the
        # window earns its speed, exactly as flex's block mask does.
        lo = tl.maximum(0, (start_m * BLOCK_M - WINDOW + 1))
        lo = (lo // BLOCK_N) * BLOCK_N
        hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)

        for start_n in range(lo, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k_ptrs = (
                K + off_b * stride_kb + off_h * stride_kh
                + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

            # FP8 tensor-core matmul, fp32 accumulate. k is (BLOCK_N, HEAD_DIM)
            # so it is transposed here to form q @ k^T.
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale

            # Causal + sliding-window mask, applied to the score tile.
            causal = offs_m[:, None] >= offs_n[None, :]
            in_window = (offs_m[:, None] - offs_n[None, :]) < WINDOW
            valid = causal & in_window & (offs_n[None, :] < N_CTX)
            qk = tl.where(valid, qk, -float("inf"))

            # Online softmax (flash-attention v2 rescaling).
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            # A fully-masked row would leave m_new = -inf and produce NaN in the
            # exponent; clamp it to 0 so such rows contribute nothing instead.
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(qk - m_safe[:, None])
            p = tl.where(valid, p, 0.0)

            alpha = tl.exp(m_i - m_safe)
            alpha = tl.where(m_i == -float("inf"), 0.0, alpha)

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v_ptrs = (
                V + off_b * stride_vb + off_h * stride_vh
                + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

            # P is in [0, 1]; scale to the top of the e4m3 range before casting
            # so the mantissa is actually used. The 1/448 is folded into
            # pv_scale by the caller. 448.0 is inlined rather than read from the
            # module-level E4M3_MAX because Triton kernels cannot close over
            # plain globals (only tl.constexpr ones).
            p_fp8 = (p * 448.0).to(tl.float8e4nv)
            acc += tl.dot(p_fp8, v, out_dtype=tl.float32)

            m_i = m_new

        acc = acc * pv_scale
        acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]

        o_ptrs = (
            Out + off_b * stride_ob + off_h * stride_oh
            + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def _per_tensor_fp8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize to e4m3 with a per-tensor amax scale. Returns (fp8, scale).

    Per-tensor rather than per-row: the same choice the linear layers made, for
    the same measured reason — on sm_120 finer-grained scaling costs back more
    than it buys (rowwise FP8 linear measured 1.01x vs bf16). `qk_norm` keeps Q
    and K bounded, which is what makes the coarse scale tolerable here.
    """
    amax = x.abs().amax().float().clamp(min=1e-12)
    scale = (amax / E4M3_MAX).item()
    return (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scale


def fp8_swa_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    *,
    block_m: int = 64,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """FP8 sliding-window causal attention forward.

    q/k/v: (B, H, T, D) bf16 or fp32. Returns (B, H, T, D) in q's dtype.
    `window`: each query attends to the `window` keys at or behind it.

    Inference/forward only — this has no backward and must not be used on a
    tensor that requires grad. Raises rather than silently producing a tensor
    with no gradient path.
    """
    if not _HAS_TRITON:
        raise RuntimeError("fp8_swa_attention_forward requires Triton")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise RuntimeError(
            "fp8_swa_attention_forward is forward-only and has no backward; "
            "calling it on tensors that require grad would silently detach them."
        )
    if q.dim() != 4:
        raise ValueError(f"expected (B, H, T, D), got {tuple(q.shape)}")

    B, H, T, D = q.shape
    if D not in (32, 64, 128):
        raise ValueError(f"head_dim must be 32/64/128 for the tl.dot tiling, got {D}")

    q_fp8, q_s = _per_tensor_fp8(q)
    k_fp8, k_s = _per_tensor_fp8(k)
    v_fp8, v_s = _per_tensor_fp8(v)

    softmax_scale = 1.0 / (D ** 0.5)
    out = torch.empty((B, H, T, D), device=q.device, dtype=q.dtype)

    grid = (triton.cdiv(T, block_m), B * H)
    _fp8_swa_fwd_kernel[grid](
        q_fp8, k_fp8, v_fp8, out,
        q_s * k_s * softmax_scale,
        v_s / E4M3_MAX,
        *q_fp8.stride(), *k_fp8.stride(), *v_fp8.stride(), *out.stride(),
        H=H, N_CTX=T, WINDOW=window, HEAD_DIM=D,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def reference_swa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int
) -> torch.Tensor:
    """Dense fp32 reference for correctness testing. O(T^2) memory — small shapes only."""
    B, H, T, D = q.shape
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = (qf @ kf.transpose(-1, -2)) / (D ** 0.5)
    idx = torch.arange(T, device=q.device)
    causal = idx[:, None] >= idx[None, :]
    in_window = (idx[:, None] - idx[None, :]) < window
    scores = scores.masked_fill(~(causal & in_window), float("-inf"))
    return (torch.softmax(scores, dim=-1) @ vf).to(q.dtype)
