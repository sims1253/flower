from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.models.attn_res import DepthRouter, routing_sites

# FlexAttention (S1) compiles the causal/local mask into the attention kernel
# without ever materializing the full T x T matrix. At seq=32K a dense mask is
# ~4GB/layer in fp32, which OOMs the 5090's 32GB; FlexAttention avoids that.
# Imported lazily so the module still loads on builds without the symbol and so
# CPU tests that never enable the flag pay no import cost.
_flex_attention: Any = None
_flex_attention_compiled: Any = None
_create_block_mask: Any = None


def _load_flex_attention() -> tuple[Any, Any]:
    """Return (flex_attention, create_block_mask), importing on first use."""
    global _flex_attention, _create_block_mask
    if _flex_attention is None:
        from torch.nn.attention.flex_attention import (
            create_block_mask as _cbm,
        )
        from torch.nn.attention.flex_attention import (
            flex_attention as _fa,
        )

        _flex_attention = _fa
        _create_block_mask = _cbm
    return _flex_attention, _create_block_mask


def _load_flex_attention_compiled() -> Any:
    """Return a torch.compiled flex_attention callable (S14-checkpoint).

    The fused CUDA flex kernel is obtained automatically when the *outer* model
    is torch.compiled, but activation checkpointing recomputes the block forward
    in a context the outer graph does not cover — so during recompute flex falls
    back to its eager ``math_attention`` path, which materializes the full
    (B, H, T, T) scores tensor and OOMs at long context (e.g. seq=8192 already
    needs ~3 GB just for scores). Compiling ``flex_attention`` as a standalone
    callable makes the fused kernel available in the recompute too, which is
    what makes activation checkpointing + FlexAttention coexist (pytorch#147879
    documents the incompatibility; pre-compiling flex is the working workaround).
    Built lazily so CPU/no-flex tests pay nothing.
    """
    global _flex_attention_compiled
    if _flex_attention_compiled is None:
        flex_attention, _ = _load_flex_attention()
        _flex_attention_compiled = torch.compile(flex_attention)
    return _flex_attention_compiled


# S14-5b: Liger FusedLinearCrossEntropy fuses the tied lm_head matmul + CE so
# the (B*T, vocab) logits tensor is never materialized during training. The
# kernel is CUDA/Triton-only. Imported lazily so CPU tests / environments
# without a working CUDA-triton path pay no import cost, mirroring
# `_load_flex_attention`. `None` = "not yet tried"; a subsequent `False` marks
# "tried and unavailable, fall back to eager".
_liger_fce_cls: Any = None


def _load_liger_fce() -> Any:
    """Return the LigerFusedLinearCrossEntropyLoss class, importing on first use.

    Returns None if Liger-Kernel is unavailable or cannot be imported; callers
    fall back to the eager head+CE path in that case. A separate runtime guard
    (CUDA-only) is applied at the call site, since the import can succeed even
    when the Triton kernel cannot run on the current device.
    """
    global _liger_fce_cls
    if _liger_fce_cls is None:
        try:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss as _FCE

            _liger_fce_cls = _FCE
        except Exception:
            _liger_fce_cls = False
    return _liger_fce_cls if _liger_fce_cls is not False else None


# S14-selective: byte-threshold policy for selective activation checkpointing.
# Activations whose storage meets this threshold are recomputed during backward
# (dropped from the saved-for-backward set); smaller ones stay materialized.
# The dominant activation in a SwiGLU/GELU block is the FFN intermediate
# (B, T, ffn_dim) — at 100M/seq8192/b4/bf16 that is ~134 MiB/layer, far above
# the threshold — so the default policy targets it while keeping the residual
# stream (B, T, d_model, ~6 MiB) and the norm outputs materialized. The
# threshold is deliberately a few MiB so it lands between those two scales.
_SELECTIVE_CKPT_BYTE_THRESHOLD = 8 * 1024 * 1024  # 8 MiB


def _selective_checkpoint_context_fn(*, threshold_bytes: int = _SELECTIVE_CKPT_BYTE_THRESHOLD) -> Any:
    """Build a ``context_fn`` for selective activation checkpointing.

    Returns a callable suitable as ``context_fn=`` to
    ``torch.utils.checkpoint.checkpoint``: a ``functools.partial`` over
    ``create_selective_checkpoint_contexts`` whose policy recomputes any op
    output whose storage is >= ``threshold_bytes`` and saves everything else.

    The policy inspects ``ctx.op_output`` (a tensor) and returns
    ``CheckpointPolicy.MUST_RECOMPUTE`` for large tensors (drop them so they
    are recomputed in backward) and ``CheckpointPolicy.MUST_SAVE`` for small
    ones (keep them materialized — no recompute). Non-tensor outputs are saved.
    Equivalent in spirit to the default byte-threshold policy described in the
    PyTorch selective-checkpointing docs, expressed as an explicit policy_fn so
    it works on any torch version that ships ``create_selective_checkpoint_contexts``.
    """
    import functools

    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    def policy_fn(ctx, op, *args, **kwargs):
        out = ctx.op_output
        if isinstance(out, torch.Tensor):
            try:
                nbytes = out.numel() * out.element_size()
            except Exception:
                nbytes = 0
            if nbytes >= threshold_bytes:
                return CheckpointPolicy.MUST_RECOMPUTE
        return CheckpointPolicy.MUST_SAVE

    return functools.partial(create_selective_checkpoint_contexts, policy_fn)


@torch._dynamo.disable
def _call_liger_fce(liger_fce: Any, lin_weight: torch.Tensor, shift_input: torch.Tensor, shift_labels: torch.Tensor):
    """Invoke Liger's fused linear+CE outside the Dynamo graph.

    WHY THE GRAPH BREAK IS MANDATORY (not an optimisation)
      Liger's kernel calls `torch.addmm(..., out_dtype=...)` internally, which
      dispatches to the `aten.addmm.dtype` overload. Inductor's lowering routes
      that to `tuned_addmm()`, which only accepts 3 positional arguments, so
      compiling through it raises

          InductorError: LoweringException: TypeError:
          tuned_addmm() takes 3 positional arguments but 4 were given

      Measured on torch 2.13.0+cu130 with and without FP8, so it is a plain
      Liger + torch.compile incompatibility on this build, not an FP8
      interaction. Before this wrapper, setting `fused_linear_ce: true`
      together with `compile_model: true` crashed the run during the first
      compile — i.e. the flag was unusable in exactly the configuration it was
      written for.

      Disabling Dynamo here costs one graph break at the very end of the
      forward, after every transformer block has already been captured, and the
      kernel it guards is hand-written Triton that gains nothing from inductor
      lowering. `torch._dynamo.disable` (rather than a config-level ban) keeps
      the memory saving available, which is the entire point of S14-5b: the
      (B*T, vocab) logits tensor is never materialised.
    """
    return liger_fce(lin_weight, shift_input, shift_labels)


def _fake_quant_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Straight-through per-tensor e4m3 quantize-dequantize.

    WHAT THIS IS FOR
      Deciding whether an FP8 attention kernel is worth building, WITHOUT
      building its backward pass first.

      `flower/kernels/fp8_swa_attention.py` implements an FP8 sliding-window
      attention forward that is measured at 1.36x flex's forward once tuned — so
      the speed is real. What is not known is whether FP8-precision attention
      degrades trained quality, and that question cannot be answered by a
      forward-only kernel. Writing the backward is a large, numerically delicate
      job to undertake on spec.

      This is the cheap substitute: the forward sees genuinely e4m3-rounded
      Q/K/V while flex still does the actual attention (so autograd works
      unchanged), and the straight-through estimator passes gradients through
      the rounding. That is the standard QAT trick, and it makes the quality
      question answerable with one ordinary training run.

    WHAT IT DOES NOT SIMULATE
      The softmax probabilities P. Those live inside flex's kernel and cannot be
      intercepted from here. A real FP8 kernel also quantizes P, which measured
      as an independent 9.4x error contributor. So this simulation is a LOWER
      BOUND on the damage: if quality degrades under this, the real kernel is
      strictly worse and the direction is closed.

    Per-tensor scaling (not per-head) deliberately: per-head measured identical
    (18.4x either way), because the error is e4m3 mantissa-limited rather than
    scale-limited.
    """
    amax = x.detach().abs().amax().clamp(min=1e-12)
    scale = amax / 448.0
    q = (x.detach() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    dq = q.to(x.dtype) * scale
    # Straight-through: forward value is dq, gradient flows to x untouched.
    return x + (dq - x).detach()


def make_causal_local_block_mask(
    local_window: int | None, seq_len: int, device: torch.device
) -> Any:
    """Build a compiled FlexAttention BlockMask for causal + optional window.

    The mask broadcasts over batch and heads (B=None, H=None). When
    `local_window` is None the mask is pure causal (full context); otherwise
    each query attends only to keys within `local_window` positions behind it.
    """
    flex_attention, create_block_mask = _load_flex_attention()

    window = local_window

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        if window is not None:
            return causal & ((q_idx - kv_idx) < window)
        return causal

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )


def _get_or_build_block_mask(module: nn.Module, seq_len: int, device: torch.device) -> Any:
    """Cached FlexAttention BlockMask for an attention module, compile-safe.

    Single source of truth for the cache logic that the per-attention-module
    ``_get_block_mask`` methods (CausalSelfAttention, FlowAttention,
    HamiltonianAttention) and the Still compact path delegate to. The cache is
    keyed on ``(local_window, seq_len)``.

    Eager mode: build+cache on miss (the legacy path — ``test_flex_block_mask_
    cache_invalidates`` pins this). Under ``torch.compile`` (``is_compiling()``
    is True): pure read — never mutate module state inside the graph, because a
    ``self._cached_block_mask = ...`` assignment captured by Dynamo aliases the
    BlockMask's internal tensors across the per-layer reads and raises
    "accessing tensor output of CUDAGraphs that has been overwritten" under
    cudagraph mode. Train.py calls ``prebuild_attention_masks`` before compiling
    so the cache is populated; a cache miss under compile is a wiring error and
    raises a clear message rather than silently mutating.
    """
    window = module.local_window
    if not torch.compiler.is_compiling():
        if (
            module._cached_block_mask is None
            or seq_len != module._cached_seq_len
            or window != module._cached_window
        ):
            module._cached_block_mask = make_causal_local_block_mask(window, seq_len, device)
            module._cached_seq_len = seq_len
            module._cached_window = window
        return module._cached_block_mask
    # Under compile: read-only. The mask must have been prebuilt (see
    # prebuild_attention_masks) so the graph only references a static object.
    if (
        module._cached_block_mask is not None
        and seq_len == module._cached_seq_len
        and window == module._cached_window
    ):
        return module._cached_block_mask
    raise RuntimeError(
        f"FlexAttention BlockMask for seq_len={seq_len}, window={window} was not "
        f"prebuilt before torch.compile. Call prebuild_attention_masks(model, "
        f"seq_len, device) in the train loop before torch.compile()."
    )


def prebuild_attention_masks(model: nn.Module, seq_len: int, device: torch.device) -> None:
    """Eagerly populate every flex-attention module's cached BlockMask.

    Call this after ``model.to(device)`` and before ``torch.compile`` so the
    compiled forward only reads the cached masks (no module-state mutation
    inside the graph — see ``_get_or_build_block_mask``). No-op for modules
    without flex enabled (``use_flex`` False/absent) and for seq lengths/windows
    already cached.
    """
    for module in model.modules():
        if getattr(module, "use_flex", False):
            _get_or_build_block_mask(module, seq_len, device)


class RMSNorm(nn.Module):
    """RMS normalization (Zhang & Sennrich, 2019) with a learnable gain.

    Cheaper than LayerNorm (no mean subtraction, no bias) and the modern default.
    Accumulates in fp32 so it stays well-behaved under bf16 autocast.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(dtype=x.dtype) * self.weight


class HeadRMSNorm(nn.Module):
    """Non-parametric RMSNorm over the head dimension, for QK normalization.

    Applied to Q and K before RoPE. Muon's updates are full-rank, so they inflate
    the spectral norms of W_Q and W_K; attention then multiplies the two in QK^T
    and the result is MaxLogit explosion (concepts/qk-stability-under-muon).
    Normalizing Q/K per head bounds the logits directly. Parameter-free, so
    enabling it does not change the parameter count.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(dtype=x.dtype)


def build_norm(config: ModelConfig, dim: int) -> nn.Module:
    """Norm layer selected by `config.norm_type`."""
    norm_type = getattr(config, "norm_type", "layernorm")
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"norm_type must be 'layernorm' or 'rmsnorm', got {norm_type!r}")


def swiglu_hidden_dim(config: ModelConfig, ffn_dim: int | None = None) -> int:
    """Effective SwiGLU hidden width for a layer of nominal width `ffn_dim`.

    SwiGLU spends three d x h matrices (gate, up, down) where GELU spends two, so
    reusing `ffn_dim` verbatim would quietly add ~50% FFN parameters and make an
    activation A/B uninterpretable. With `ffn_param_match` the width is scaled to
    2/3 * ffn_dim, rounded to a multiple of 64 for kernel friendliness.
    """
    nominal = config.ffn_dim if ffn_dim is None else ffn_dim
    if not getattr(config, "ffn_param_match", True):
        return nominal
    target = (nominal * 2) // 3
    return max(64, round(target / 64) * 64)


def layer_attn_windows(config: ModelConfig) -> list[int | None]:
    """Per-layer attention windows, honouring `attn_window_schedule`.

    `None` in the returned list means that layer attends over the full context.
    """
    schedule = getattr(config, "attn_window_schedule", None)
    if not schedule:
        return [config.local_window] * config.num_layers
    if len(schedule) != config.num_layers:
        raise ValueError(
            f"attn_window_schedule has {len(schedule)} entries but num_layers is "
            f"{config.num_layers}; they must match."
        )
    return [None if w is None else int(w) for w in schedule]


def layer_ffn_dims(config: ModelConfig) -> list[int]:
    """Per-layer nominal FFN widths, honouring `ffn_dim_schedule`."""
    schedule = getattr(config, "ffn_dim_schedule", None)
    if not schedule:
        return [config.ffn_dim] * config.num_layers
    if len(schedule) != config.num_layers:
        raise ValueError(
            f"ffn_dim_schedule has {len(schedule)} entries but num_layers is "
            f"{config.num_layers}; they must match."
        )
    return [int(d) for d in schedule]


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def causal_mask(seq_len: int, device: torch.device, local_window: int | None = None) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    mask = idx[:, None] >= idx[None, :]
    if local_window is not None:
        mask &= (idx[:, None] - idx[None, :]) < local_window
    return mask


def scaled_dot_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


class RotaryEmbedding(nn.Module):
    """Standard RoPE with cached cos/sin tables."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim")
        self._base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer(
            "cos",
            freqs.cos().repeat_interleave(2, dim=-1).view(1, 1, max_seq_len, head_dim),
            persistent=False,
        )
        self.register_buffer(
            "sin",
            freqs.sin().repeat_interleave(2, dim=-1).view(1, 1, max_seq_len, head_dim),
            persistent=False,
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _extend_cache(self, seq_len: int) -> None:
        """Extend RoPE buffers to cover seq_len (called lazily when eval exceeds training length)."""
        head_dim = self.get_buffer("cos").shape[-1]
        inv_freq = 1.0 / (self._base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos().repeat_interleave(2, dim=-1).view(1, 1, seq_len, head_dim), persistent=False)
        self.register_buffer("sin", freqs.sin().repeat_interleave(2, dim=-1).view(1, 1, seq_len, head_dim), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos_cache = self.get_buffer("cos")
        sin_cache = self.get_buffer("sin")
        if seq_len > cos_cache.shape[-2]:
            self._extend_cache(seq_len)
            cos_cache = self.get_buffer("cos")
            sin_cache = self.get_buffer("sin")
        cos = cos_cache[..., :seq_len, :].to(device=q.device, dtype=q.dtype)
        sin = sin_cache[..., :seq_len, :].to(device=q.device, dtype=q.dtype)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


class FeedForward(nn.Module):
    """GELU MLP (legacy default) or SwiGLU, selected by `config.ffn_activation`.

    `config` is optional so the many variant modules that construct
    `FeedForward(d_model, ffn_dim, dropout)` positionally keep working and keep
    their GELU behaviour.
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        dropout: float = 0.0,
        config: ModelConfig | None = None,
        ffn_dim_override: int | None = None,
    ) -> None:
        super().__init__()
        activation = getattr(config, "ffn_activation", "gelu") if config is not None else "gelu"
        bias = bool(getattr(config, "use_bias", True)) if config is not None else True
        if activation not in {"gelu", "swiglu"}:
            raise ValueError(f"ffn_activation must be 'gelu' or 'swiglu', got {activation!r}")
        self.activation = activation
        # S10: per-token scale on the SwiGLU `up` projection so the gated
        # product does not amplify outliers under FP8/FP4. Mathematically
        # equivalent (w_down is linear, so the scale cancels); the only effect
        # is a smaller dynamic range on the intermediate fused tensor. Set for
        # both branches so the attribute always exists; the GELU path ignores it.
        self.smooth_swiglu = bool(getattr(config, "smooth_swiglu", False)) if config is not None else False
        if activation == "gelu":
            # Keep the original nn.Sequential layout: the parameter names
            # (net.0.weight / net.3.weight) are baked into every checkpoint
            # written before sweep 13, including the phase-0 bases that phase-1
            # loads via still_pretrained_base.
            self.net = nn.Sequential(
                nn.Linear(d_model, ffn_dim, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model, bias=bias),
            )
            self.net[-1]._is_residual_out = True
        else:
            hidden = swiglu_hidden_dim(config, ffn_dim_override) if config is not None else ffn_dim
            self.hidden_dim = hidden
            self.dropout = nn.Dropout(dropout)
            self.gate = nn.Linear(d_model, hidden, bias=bias)
            self.up = nn.Linear(d_model, hidden, bias=bias)
            self.down = nn.Linear(hidden, d_model, bias=bias)
            self.down._is_residual_out = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return self.net(x)
        gate = self.gate(x)
        up = self.up(x)
        act = F.silu(gate)
        if self.smooth_swiglu:
            # Per-token scale on `up`: prevents SwiGLU outlier amplification
            # under FP8/FP4 (S10). w_down is linear, so the scale cancels and
            # the result is mathematically identical to the standard path.
            up_scale = up.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
            fused = self.dropout(act * (up / up_scale))
            out = self.down(fused)
            return out * up_scale
        fused = self.dropout(act * up)
        return self.down(fused)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, local_window: int | None = None) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.local_window = local_window
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, base=config.rope_base)
        bias = bool(getattr(config, "use_bias", True))
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=bias)
        self.out = nn.Linear(config.d_model, config.d_model, bias=bias)
        self.out._is_residual_out = True  # tagged for depth-scaled init
        # Parameter-free QK normalization; see HeadRMSNorm.
        self.qk_norm = HeadRMSNorm() if bool(getattr(config, "qk_norm", False)) else None
        # S2.4: optional RBF distance kernel as additive attention bias.
        # rbf bias = -scale * (i-j)^2 / window^2, learnable scale, per-head.
        self.kernel_bias = getattr(config, "local_attn_kernel_bias", "none")
        if self.kernel_bias not in {"none", "rbf"}:
            raise ValueError("local_attn_kernel_bias must be 'none' or 'rbf'")
        if self.kernel_bias == "rbf":
            init_scale = float(getattr(config, "local_attn_rbf_scale", 4.0))
            self.rbf_log_scale = nn.Parameter(torch.full((self.num_heads,), math.log(max(init_scale, 1e-3))))
        # S1 (FlexAttention). Opt-in; the SDPA path stays the default so the
        # existing (non-flex) tests and published runs reproduce bit-for-bit.
        self.use_flex = bool(getattr(config, "flex_attention", False))
        # FP8-attention quality probe. Rounds Q/K/V to e4m3 in the forward with a
        # straight-through gradient, so the cost of FP8-precision attention can
        # be measured in a normal training run before any FP8 attention kernel
        # backward is written. See _fake_quant_e4m3 for the full rationale and
        # for what it does NOT simulate. Off by default; this is a measurement
        # instrument, not a speedup (it is slightly SLOWER than plain bf16).
        self.fp8_attention_sim = bool(getattr(config, "fp8_attention_sim", False))
        # S14-checkpoint: when activation checkpointing wraps the block forward,
        # flex must use its standalone-compiled fused kernel so the recompute
        # (which the outer torch.compile graph does not cover) still runs fused
        # instead of falling back to the OOM-prone dense math path. See
        # _load_flex_attention_compiled.
        self._flex_needs_compile = self.use_flex and bool(
            getattr(config, "activation_checkpoint", False)
        )
        # Block-mask cache: create_block_mask is expensive (it compiles). The
        # mask only changes when seq_len or local_window changes, so cache it.
        self._cached_block_mask: Any = None
        self._cached_seq_len: int = 0
        self._cached_window: int | None = None

    def _get_block_mask(self, seq_len: int, device: torch.device) -> Any:
        # Delegates to the shared, compile-safe cache logic. See
        # _get_or_build_block_mask for why the mutation must not happen under
        # torch.compile (cudagraph aliasing of the cached BlockMask).
        return _get_or_build_block_mask(self, seq_len, device)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _rbf_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Normalised positions so behaviour is window-size invariant.
        denom = float(self.local_window) if self.local_window is not None else float(seq_len)
        idx = torch.arange(seq_len, device=device, dtype=dtype) / max(denom, 1.0)
        d2 = (idx[:, None] - idx[None, :]).pow(2)  # (T, T)
        scale = self.rbf_log_scale.exp().to(device=device, dtype=dtype).view(self.num_heads, 1, 1)
        return (-scale * d2).unsqueeze(0)  # (1, H, T, T)

    def qkv_heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to per-head (q, k, v), apply QK-norm, then RoPE.

        Single source of truth for the pre-attention pipeline. `still_lm.py`
        re-implements the block forward in order to intercept the KV cache, so
        it must call this rather than composing qkv/rope itself — otherwise
        QK-norm would silently apply in the base model's own forward but not in
        the Still teacher/student passes, and a base pretrained one way would be
        evaluated the other.

        Returns q, k, v each shaped (B, H, T, head_dim).
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        if self.qk_norm is not None:
            q, k = self.qk_norm(q), self.qk_norm(k)
        q, k = self.rope(q, k)
        if self.fp8_attention_sim:
            # Quantize AFTER QK-norm and RoPE, i.e. exactly the tensors a real
            # FP8 attention kernel would consume. See _fake_quant_e4m3.
            q, k, v = _fake_quant_e4m3(q), _fake_quant_e4m3(k), _fake_quant_e4m3(v)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_heads(x)
        seq_len = x.shape[1]
        if self.use_flex:
            return self._forward_flex(x, q, k, v, seq_len)
        return self._forward_sdpa(x, q, k, v, seq_len)

    def _forward_sdpa(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int) -> torch.Tensor:
        # Use fused scaled-dot-product attention (flash / mem-efficient kernels).
        # The previous path materialized a dense (B, H, T, T) fp32 score tensor per
        # layer; at B16/T2048 that is ~1.6 GB/layer and, with all layers' backward
        # activations, overflowed the 32 GB card. On WSL2 the overflow does not OOM
        # — it silently spills into shared host RAM over PCIe and throughput
        # collapses. SDPA never materializes the T x T scores, so it stays on-card.
        keep = causal_mask(seq_len, x.device, self.local_window).view(1, 1, seq_len, seq_len)
        if self.kernel_bias == "rbf":
            # RBF needs an additive per-head bias; fold the causal/local mask into
            # the same float attn_mask so SDPA still fuses the softmax+matmul.
            attn_mask = self._rbf_bias(seq_len, x.device, q.dtype).expand(1, self.num_heads, seq_len, seq_len).clone()
            attn_mask = attn_mask.masked_fill(~keep, torch.finfo(q.dtype).min)
        else:
            attn_mask = keep
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(x.shape)
        return self.out(out)

    def _forward_flex(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int) -> torch.Tensor:
        # S1: FlexAttention compiles the mask into the kernel. The block mask is
        # cached (see _get_block_mask); RBF bias becomes a score_mod so no dense
        # (B, H, T, T) tensor is ever materialized. The fused CUDA kernel is
        # obtained automatically when the outer model is torch.compiled; in eager
        # mode flex_attention still runs (unfused) — correct, just not fast.
        # S14-checkpoint: when activation checkpointing is on, use the
        # standalone-compiled flex so the recompute forward also gets the fused
        # kernel (otherwise it OOMs materializing the dense scores).
        if self._flex_needs_compile:
            flex_attention = _load_flex_attention_compiled()
        else:
            flex_attention, _ = _load_flex_attention()
        block_mask = self._get_block_mask(seq_len, q.device)
        if self.kernel_bias == "rbf":
            # Per-head RBF scale; close over the current parameter value so the
            # schedule stays differentiable through rbf_log_scale.
            rbf_scale = self.rbf_log_scale.exp().to(device=q.device, dtype=q.dtype)
            denom_sq = float(self.local_window) ** 2 if self.local_window is not None else float(seq_len) ** 2

            def rbf_score_mod(score: torch.Tensor, b, h, q_idx, kv_idx) -> torch.Tensor:
                dist = (q_idx - kv_idx).float()
                return score - rbf_scale[h] * (dist ** 2) / denom_sq

            out = flex_attention(q, k, v, score_mod=rbf_score_mod, block_mask=block_mask)
        else:
            out = flex_attention(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).contiguous().view(x.shape)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        attention: nn.Module | None = None,
        ffn_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = build_norm(config, config.d_model)
        self.attn = attention or CausalSelfAttention(config, config.local_window)
        self.ln2 = build_norm(config, config.d_model)
        # `ffn_dim` overrides config.ffn_dim for this layer only (TLM taper).
        self.ff = FeedForward(
            config.d_model,
            config.ffn_dim if ffn_dim is None else ffn_dim,
            config.dropout,
            config=config,
            ffn_dim_override=ffn_dim,
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, memory


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(blocks)
        self.ln = build_norm(config, config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        # S3: FP8 matmul for the lm_head projection (Blackwell/Hopper). Off by
        # default; guarded again in _compute_logits by dtype + CUDA checks.
        self.fp8_lm_head = bool(getattr(config, "fp8_lm_head", False))
        # S4: compute the next-token cross-entropy in BF16 instead of FP32.
        self.bf16_cross_entropy = bool(getattr(config, "bf16_cross_entropy", False))
        # S14-5b: Liger FusedLinearCrossEntropy. Fuses the tied lm_head matmul +
        # CE so the (B*T, vocab) logits tensor is never materialized during
        # training — the binding memory constraint at long seq / large vocab.
        # Training-time only; _compute_logits (eager, incl. FP8-head eval) is
        # untouched. Constructed lazily: the Liger class is CUDA/Triton-only, so
        # defer to first use and fall back to eager if it is unavailable or the
        # device is not CUDA. See `_fused_cross_entropy`.
        self.fused_linear_ce = bool(getattr(config, "fused_linear_ce", False))
        self._liger_fce: Any = None
        # S14-checkpoint: activation checkpointing on the transformer blocks.
        # Stored verbatim (False | True | "selective") so forward can branch on
        # the mode. bool() would collapse "selective" to True and lose the mode.
        self.activation_checkpoint = getattr(config, "activation_checkpoint", False)
        # S8: auxiliary Multi-Token-Prediction heads for t+2, t+3, ... These are
        # untied heads on the already-normed hidden state; the main tied head
        # still predicts t+1. None when mtp_extra_heads <= 0 (legacy default).
        self.mtp_extra_heads = int(getattr(config, "mtp_extra_heads", 0))
        self.mtp_weight = float(getattr(config, "mtp_weight", 0.5))
        if self.mtp_extra_heads > 0:
            self.mtp_heads = nn.ModuleList([
                nn.Linear(config.d_model, config.vocab_size, bias=False)
                for _ in range(self.mtp_extra_heads)
            ])
        else:
            self.mtp_heads = None
        # Depth-axis routing (AttnRes family). None unless explicitly enabled.
        self.attn_res_sites: list[int] = []
        self.depth_router: DepthRouter | None = None
        attn_res = getattr(config, "attn_res", "none")
        if attn_res not in {"none", "delta_block"}:
            raise ValueError(f"attn_res must be 'none' or 'delta_block', got {attn_res!r}")
        if attn_res == "delta_block":
            if max(1, getattr(config, "loop_count", 1)) > 1:
                raise ValueError(
                    "attn_res=delta_block is incompatible with loop_count>1: looping "
                    "re-enters the same blocks, so block deltas are not well defined."
                )
            self.attn_res_sites = routing_sites(len(blocks), int(getattr(config, "attn_res_blocks", 8)))
            self.depth_router = DepthRouter(
                config.d_model,
                num_sites=len(self.attn_res_sites),
                key_mode=getattr(config, "attn_res_key", "full"),
                rank=int(getattr(config, "attn_res_rank", 64)),
            )
        # Diagnostics are constant per model or require a full module walk; both
        # were previously recomputed on every forward. Cache the constants and
        # make the walk opt-out so it stays off inside a compiled graph.
        self.collect_module_diagnostics = True
        self._static_diagnostics: dict[str, Any] | None = None
        init_scheme = getattr(config, "init_scheme", "torch")
        if init_scheme not in {"torch", "scaled"}:
            raise ValueError(f"init_scheme must be 'torch' or 'scaled', got {init_scheme!r}")
        # S12.2: orthogonal weight init is a separate, orthogonal flag. It wins
        # over the generic scaled scheme when both are set (its residual-output
        # scaling already mirrors _apply_scaled_init). With neither requested
        # the module keeps whatever init its submodules ran in __init__.
        if getattr(config, "orthogonal_init", False):
            self._apply_orthogonal_init()
        elif init_scheme == "scaled":
            self._apply_scaled_init()

    def _apply_scaled_init(self) -> None:
        """GPT-2 style initialisation.

        Linear and Embedding weights are drawn from N(0, init_std) and biases
        zeroed; residual output projections (attention `out`, FFN `down`) are
        additionally scaled by 1/sqrt(2 * num_layers) so the variance added to
        the residual stream does not grow with depth.

        Runs after any submodule-specific init (e.g. zero-init velocity heads),
        so those are re-drawn — enable this only on variants where the generic
        scheme is what you want, which is the vanilla base and its AttnRes arms.
        """
        std = float(getattr(self.config, "init_std", 0.02))
        depth_scale = 1.0 / math.sqrt(2 * max(1, len(self.blocks)))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                scale = std * (depth_scale if getattr(module, "_is_residual_out", False) else 1.0)
                nn.init.normal_(module.weight, mean=0.0, std=scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        # The DepthRouter must stay an exact identity at init; the generic walk
        # above does not touch its bare Parameters, but re-assert it explicitly.
        if self.depth_router is not None:
            nn.init.zeros_(self.depth_router.query)
            nn.init.zeros_(self.depth_router.gate)

    def _apply_orthogonal_init(self) -> None:
        """Orthogonal weight initialisation for 2D matrices (Parameter Golf).

        Synergises with Muon: the Newton-Schulz step preserves orthogonality,
        so starting orthogonal keeps the optimizer in its ideal regime. Residual
        output projections (attention `out`, FFN `down`) are scaled by
        1/sqrt(2*num_layers) like the scaled scheme. Biases zeroed. Embeddings
        keep a small normal init so the tied lm_head does not start at 1.
        """
        depth_scale = 1.0 / math.sqrt(2 * max(1, len(self.blocks)))
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight.ndim == 2:
                nn.init.orthogonal_(module.weight)
                if getattr(module, "_is_residual_out", False):
                    module.weight.data *= depth_scale
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=float(getattr(self.config, "init_std", 0.02)),
                )
        if self.depth_router is not None:
            nn.init.zeros_(self.depth_router.query)
            nn.init.zeros_(self.depth_router.gate)

    def _compute_logits(self, x_normed: torch.Tensor) -> torch.Tensor:
        """LM head projection on an already-normed hidden state (S3).

        When `fp8_lm_head` is set and the activations are BF16 on CUDA, the
        tied-head matmul runs via `torch._scaled_mm` in float8_e4m3fn
        (Blackwell/Hopper). Otherwise this is the plain `self.head` matmul.

        FP8 is used in EVAL/INFERENCE only: `torch._scaled_mm` has no backward
        kernel in current PyTorch, so during training (when the head logits feed
        the loss and need gradients) we fall back to the BF16 head. This matches
        how production FP8 heads are used — the expensive logits matmul runs in
        FP8 for eval, and is recomputed in BF16 for the training loss/backward.
        """
        if (
            self.fp8_lm_head
            and not self.training
            and x_normed.dtype == torch.bfloat16
            and x_normed.is_cuda
        ):
            return self._fp8_head(x_normed)
        return self.head(x_normed)

    def _fp8_head(self, x_normed: torch.Tensor) -> torch.Tensor:
        """FP8 matmul for the lm_head projection via torch._scaled_mm (S3).

        Requires BF16 activations + CUDA (Blackwell/Hopper). Computes per-row
        amax scales, casts to float8_e4m3fn, runs `_scaled_mm`, casts logits
        back to BF16. The cast is on a view of the tied embedding weight -
        master weights stay BF16.
        """
        B, T, D = x_normed.shape
        x2d = x_normed.reshape(-1, D)  # (B*T, D)
        w = self.head.weight  # (vocab, D), tied to the embedding
        x_fp8 = x2d.to(torch.float8_e4m3fn)
        w_fp8 = w.to(torch.float8_e4m3fn)
        # Row-wise per-tensor scaling (amax / FP8_E4M3 max = 448.0). _scaled_mm
        # requires float32 scales; for row-wise scaling scale_a is (M, 1) and
        # scale_b is (1, N) matching the output (M, N) = (B*T, vocab) shape.
        scale_a = (x2d.amax(dim=-1, keepdim=True).float() / 448.0)  # (B*T, 1)
        scale_b = (w.amax(dim=-1, keepdim=True).t().contiguous().float() / 448.0)  # (1, vocab)
        logits2d = torch._scaled_mm(
            x_fp8, w_fp8.t(),
            scale_a=scale_a, scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )
        return logits2d.view(B, T, -1)

    def _cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Shifted next-token CE; optionally in BF16 (S4)."""
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        if self.bf16_cross_entropy:
            shift_logits = shift_logits.to(torch.bfloat16)
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

    def _ensure_liger_fce(self) -> bool:
        """Lazily build the Liger FusedLinearCrossEntropy loss module (S14-5b).

        Returns True iff the fused path can run on the current device: the
        Liger class imported successfully AND the model's parameters live on
        CUDA (the underlying kernel is Triton/CUDA-only). Constructs the
        stateless loss module once on first success and caches it. Returns
        False (so `forward` falls back to eager) if Liger is unavailable or the
        device is not CUDA. Emits a one-time warning on the first such fallback
        so a user who set `fused_linear_ce=True` does not silently run eager.
        """
        if not self.fused_linear_ce:
            return False
        if self._liger_fce is not None:
            return True
        cls = _load_liger_fce()
        # The Triton kernel requires CUDA. Resolve via the embedding parameter
        # rather than an input tensor so this is callable before forward.
        on_cuda = self.token.weight.is_cuda
        if cls is None or not on_cuda:
            if not getattr(self, "_warned_fused_ce_fallback", False):
                reason = "liger-kernel is unavailable" if cls is None else "the model is not on CUDA"
                import warnings

                warnings.warn(
                    f"fused_linear_ce=True is set but {reason}; falling back to the eager "
                    f"lm_head + cross-entropy path (no memory saving). Set fused_linear_ce=False "
                    f"to silence this, or move the model to CUDA.",
                    stacklevel=2,
                )
                self._warned_fused_ce_fallback = True
            return False
        self._liger_fce = cls()
        return True

    def _multi_hot_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Multi-hot CE for Token Superposition Training phase 1 (S9).

        In TST phase 1 each position holds a BAG of `s` consecutive tokens whose
        embeddings were averaged on the way in, so the natural target for
        position i is the SET of `s` tokens in bag i+1 — not a single token.
        The objective is the mean negative log-likelihood over that set:

            L = -1/(L*s) * sum_i sum_{t in bag_{i+1}} log p(t | bag_{<=i})

        which is exactly cross-entropy against a uniform multi-hot target. It is
        computed by gathering the `s` target log-probabilities rather than
        materialising a (B, L, V) one-hot tensor, and rather than repeating the
        logits `s` times — at vocab 16384 either of those would dominate the
        memory this technique exists to save.

        `logits` is (B, L, V) and `labels` is (B, L, s), both already at the
        COMPRESSED length L = T/s. The usual next-token shift applies at the bag
        level: bag i predicts bag i+1.
        """
        if labels.size(1) < 2:
            # Fewer than two bags: nothing to predict. Return a real zero that
            # still carries grad_fn so the backward graph stays well-formed.
            return logits.sum() * 0.0
        shift_logits = logits[:, :-1]  # (B, L-1, V)
        shift_labels = labels[:, 1:]  # (B, L-1, s)
        logp = torch.log_softmax(shift_logits.float(), dim=-1)
        picked = logp.gather(-1, shift_labels)  # (B, L-1, s)
        return -picked.mean()

    def _fused_cross_entropy(
        self,
        x_normed: torch.Tensor,
        labels: torch.Tensor,
        lin_weight: torch.Tensor,
        *,
        offset: int = 1,
    ) -> torch.Tensor:
        """Fused linear + cross-entropy on a pre-normed hidden state (S14-5b).

        Liger's `LigerFusedLinearCrossEntropyLoss` takes the projection weight
        and the pre-projection activations directly and fuses the matmul + CE
        internally, so the `(B*T, vocab)` logits tensor is never materialized.
        It takes the weight *by reference*, so passing the tied embedding
        weight keeps the tying load-bearing: the embedding gradient flows back
        through it as before.

        The shift that the eager path applies *after* projecting
        (`logits[:, :-1]`, `labels[:, 1:]`) must instead be applied to the
        activations and labels *before* the fused call, since the fused kernel
        projects internally. `offset` generalises the shift so the same helper
        serves the main head (offset=1, predicts t+1) and each MTP head
        (offset=i+2, predicts t+2, t+3, ...); the aligned slices are
        `x_normed[:, :T-offset]` -> `labels[:, offset:]`.

        Liger accumulates the softmax/CE internally in fp32 regardless of input
        dtype, so `bf16_cross_entropy` (S4) is a no-op under this path: the
        fused kernel is at least as precise as eager bf16 CE (it upcasts), so a
        user who disabled bf16 CE for numerical reasons is still served a
        high-precision reduction. CUDA/Triton-only; `forward` checks
        availability + device before calling this.
        """
        T = x_normed.size(1)
        if offset >= T or offset >= labels.size(1):
            # Nothing to predict at this offset (sequence shorter than the
            # horizon); mirror the eager MTP path's guard with a zero loss.
            return x_normed.new_zeros(())
        shift_input = x_normed[:, : T - offset].reshape(-1, x_normed.size(-1))
        shift_labels = labels[:, offset:].reshape(-1)
        return _call_liger_fce(self._liger_fce, lin_weight, shift_input, shift_labels)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, Any]:
        # Allow seq_len > max_seq_len during eval (Sweep 7 A1: eval_seq_len > max_seq_len).
        # RoPE cache extends lazily; local attention mask is computed on-the-fly.
        #
        # S9 / Token Superposition Training: a 3-D `input_ids` of shape
        # (B, T/s, s) is a BAGGED batch — each position holds `s` consecutive
        # token ids whose embeddings are averaged into one superposed vector.
        # The transformer then runs over T/s positions instead of T, which is
        # where TST's speedup comes from. 2-D input is the ordinary NTP path.
        bagged = input_ids.dim() == 3
        if bagged:
            # Mean over the bag axis. `token` is (V, D), so indexing with a 3-D
            # id tensor yields (B, T/s, s, D); averaging gives (B, T/s, D).
            x = self.token(input_ids).mean(dim=2)
        else:
            x = self.token(input_ids)
        memory = None
        diagnostics: dict[str, Any] = {}
        # A2 looped transformer: run the block stack `loop_count` times. Memory state
        # carries across loops, so the bank acts as a scratchpad / persistent state.
        # loop_count=1 reproduces the standard single-pass behaviour.
        loops = max(1, getattr(self.config, "loop_count", 1))
        if self.depth_router is None:
            # S14-checkpoint: activation checkpointing on each block during
            # training. Activations are not retained for backward (recomputed
            # from the block inputs instead), cutting activation memory from
            # O(num_layers) to O(1) at the cost of one extra forward per
            # backward. use_reentrant=False preserves the differentiable memory
            # tensor carried between blocks and saves/restores RNG state so
            # dropout is identical across the two passes (verified by a
            # same-seed loss-identity test). Skipped in eval (no backward, so
            # checkpointing only wastes the recompute) and when the flag is off.
            # Loop count > 1 is left uncheckpointed: the shared memory state
            # across loops makes per-loop checkpointing interact subtly with
            # the carried memory, and loop_count>1 is not used in the
            # long-context configs this targets.
            #
            # activation_checkpoint values:
            #   False  -> off (legacy)
            #   True   -> full checkpointing (recompute every activation)
            #   "selective" -> recompute only large activations (FFN
            #     intermediate); keep cheap ones (residual stream) materialized.
            #     Uses create_selective_checkpoint_contexts (torch 2.13) with a
            #     byte-threshold policy via context_fn. The flex-compile
            #     workaround still applies (selective also recomputes flex).
            do_ckpt = (
                self.training
                and self.activation_checkpoint
                and loops == 1
            )
            if do_ckpt:
                from torch.utils.checkpoint import checkpoint

                if self.activation_checkpoint == "selective":
                    # Selective: build the byte-threshold context_fn once
                    # (it is stateless per call) and reuse across layers.
                    context_fn = _selective_checkpoint_context_fn()
                    for block in self.blocks:
                        x, memory = checkpoint(
                            block, x, memory,
                            use_reentrant=False,
                            context_fn=context_fn,
                        )
                elif self.activation_checkpoint == "ffn":
                    # FFN-only: checkpoint ONLY the feed-forward sub-layer of
                    # each vanilla block, keeping attention (Q/K/V/O) live.
                    # FlashAttention/Flex already recomputes the O(T^2) score
                    # matmul in backward, so the only thing full checkpointing
                    # buys on the attention side is dropping Q/K/V — which are
                    # large at long T but cheap to keep. The FFN intermediate is
                    # both the biggest single activation AND cheap to recompute
                    # (dense matmuls), making it the efficient checkpoint target.
                    # Measured: ~37% memory saving for ~10% throughput cost, vs
                    # full's ~50% saving for ~100% cost (see config docstring).
                    # Vanilla blocks only; memory-variant blocks fall back to
                    # full (their forward interleaves memory ops with attention,
                    # so there is no clean FFN-only boundary).
                    for block in self.blocks:
                        if isinstance(block, TransformerBlock):
                            attn_out = block.attn(block.ln1(x))
                            x = x + attn_out
                            x = x + checkpoint(block.ff, block.ln2(x), use_reentrant=False)
                            # memory is unchanged by vanilla blocks (passed through).
                        else:
                            if not getattr(self, "_warned_ffn_fallback", False):
                                print(
                                    "[checkpoint] activation_checkpoint='ffn' is only "
                                    "supported on vanilla blocks; falling back to full "
                                    "checkpointing for memory-variant blocks."
                                )
                                self._warned_ffn_fallback = True
                            x, memory = checkpoint(block, x, memory, use_reentrant=False)
                else:
                    for block in self.blocks:
                        # memory=None on the first block; checkpoint requires at
                        # least one input that requires grad, which `x` (coming
                        # from the embedding) does under training. Passing
                        # use_reentrant=False avoids the reentrant variant's RNG
                        # and grad-input quirks and handles memory=None correctly.
                        x, memory = checkpoint(block, x, memory, use_reentrant=False)
            else:
                for _ in range(loops):
                    for block in self.blocks:
                        x, memory = block(x, memory)
        else:
            # Delta Block AttnRes: sources are the embedding plus one delta per
            # completed block. At each site the router additively mixes the
            # routed sources back into the stream, then the next delta is
            # measured from the post-routing state.
            sources = [x]
            prev = x
            site = 0
            for layer_idx, block in enumerate(self.blocks):
                x, memory = block(x, memory)
                if site < len(self.attn_res_sites) and layer_idx == self.attn_res_sites[site]:
                    sources.append(x - prev)
                    x = self.depth_router(site, x, sources)
                    prev = x
                    site += 1
        x_normed = self.ln(x)
        loss = None
        # S14-5b: fused lm_head + CE path. Training-time only, CUDA/Triton only.
        # When active the (B*T, vocab) logits tensor is never materialized, so
        # `logits` stays None and every eval/logprob consumer (which runs in
        # eval mode or without labels) keeps the eager path below. Falls back to
        # eager automatically if Liger is unavailable or the device is not CUDA.
        # S9 TST: bagged labels are (B, T/s, s) and need the multi-hot objective,
        # which neither the eager nor the Liger CE path implements. Handled first
        # so the flags below cannot route a bagged batch into a 2-D loss.
        if labels is not None and labels.dim() == 3:
            logits = self._compute_logits(x_normed)
            loss = self._multi_hot_cross_entropy(logits, labels)
            return {"logits": logits, "loss": loss, "memory": memory, "diagnostics": diagnostics}

        use_fused_ce = (
            self.fused_linear_ce
            and self.training
            and labels is not None
            and self._ensure_liger_fce()
        )
        if use_fused_ce:
            # Main tied head: predict t+1 from x_normed. The tied weight is
            # passed by reference so the embedding gradient flows back through it.
            loss = self._fused_cross_entropy(x_normed, labels, self.token.weight, offset=1)
            # S8 MTP heads: each untied head predicts t+2, t+3, ... The fused
            # path applies the per-head shift inside `_fused_cross_entropy`.
            if self.mtp_heads is not None:
                for i, mtp_head in enumerate(self.mtp_heads):
                    mtp_loss = self._fused_cross_entropy(
                        x_normed, labels, mtp_head.weight, offset=i + 2
                    )
                    loss = loss + self.mtp_weight * mtp_loss
            logits = None
        else:
            logits = self._compute_logits(x_normed)
            if labels is not None:
                loss = self._cross_entropy(logits, labels)
                # S8: auxiliary Multi-Token-Prediction heads. Each untied head on
                # the already-normed hidden state predicts t+2, t+3, ... The main
                # tied head above still handles the t+1 prediction. `x_normed` is
                # reused verbatim - the FP8 path only affects the tied matmul, the
                # MTP heads are always plain BF16 Linears.
                #
                # `self.training` GUARD IS LOAD-BEARING. MTP is a TRAINING-ONLY
                # auxiliary objective: at eval we want the quality of the t+1
                # head alone, which is what val_bpb is compared across arms.
                # Without this guard the eval loss silently became
                # `main + mtp_weight * sum(aux)`, because the fused path above is
                # itself gated on `self.training` and so eval always falls into
                # THIS branch. The first MTP screen measured val_bpb 1.128 ->
                # 2.036 (1 head) -> 3.069 (2 heads), which reads as catastrophic
                # divergence but is entirely the leak: those ratios imply t+2 and
                # t+3 losses of 1.61x and 1.83x the main loss, monotone in offset
                # exactly as predicting further ahead should be. Guarded by
                # tests/test_mtp_eval_loss.py.
                if self.mtp_heads is not None and self.training:
                    for i, mtp_head in enumerate(self.mtp_heads):
                        offset = i + 2  # predict t+2, t+3, ...
                        mtp_logits = mtp_head(x_normed)
                        # Align: mtp_logits[:, :T-offset] predicts labels[:, offset:].
                        if offset < mtp_logits.size(1) and offset < labels.size(1):
                            mtp_logits = mtp_logits[:, : mtp_logits.size(1) - offset]
                            mtp_labels = labels[:, offset:]
                            if self.bf16_cross_entropy:
                                mtp_logits = mtp_logits.to(torch.bfloat16)
                            mtp_loss = F.cross_entropy(
                                mtp_logits.reshape(-1, mtp_logits.size(-1)),
                                mtp_labels.reshape(-1),
                            )
                            loss = loss + self.mtp_weight * mtp_loss
        if self._static_diagnostics is None:
            self._static_diagnostics = {
                "parameter_count": count_parameters(self),
                "config": asdict(self.config),
            }
        diagnostics.update(self._static_diagnostics)
        # Generic diagnostic walker. Any submodule can stash a scalar by
        # setting `self.last_diag_<field> = float(...)` in its forward; we
        # aggregate across modules (mean and max) and emit as
        # `<field>_mean` / `<field>_max`. Tensor buffers in the same naming
        # convention (0-d) are also picked up so symplectic flows can register
        # buffers without per-instance Python attrs.
        collected: dict[str, list[float]] = {}
        modules = self.modules() if self.collect_module_diagnostics else ()
        for module in modules:
            for attr_name in dir(module):
                if not attr_name.startswith("last_diag_"):
                    continue
                try:
                    value = getattr(module, attr_name)
                except AttributeError:
                    continue
                key = attr_name[len("last_diag_"):]
                if isinstance(value, torch.Tensor):
                    if value.ndim != 0:
                        continue
                    collected.setdefault(key, []).append(float(value.detach().cpu()))
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    collected.setdefault(key, []).append(float(value))
        for key, values in collected.items():
            t = torch.tensor(values)
            diagnostics[f"{key}_mean"] = float(t.mean())
            diagnostics[f"{key}_max"] = float(t.max())
        return {"logits": logits, "loss": loss, "diagnostics": diagnostics}
