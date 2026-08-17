from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig


# ---------------------------------------------------------------------------
# Causal-memory primitives (config.causal_memory=True).
#
# The legacy memory write aggregates the WHOLE window into a single (B, S, D)
# bank, so the bank a layer-i+1 read consumes at position t contains tokens
# AFTER t (see ModelConfig.causal_memory). These helpers compute the causal
# counterparts: per-position prefix reductions of (B, T, D) tensors and a
# prefix-softmax cross-attention. Everything is differentiable, adds no
# parameters, and is only ever called from the causal branch of a variant's
# write path — the legacy (flag-off) ops are untouched.
# ---------------------------------------------------------------------------


def causal_running_mean(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Prefix mean along ``dim``: out[..., t, ...] = mean(x[..., :t+1], ...).

    Causal counterpart of ``x.mean(dim=dim)`` via cumsum / prefix count.
    """
    total = torch.cumsum(x, dim=dim)
    n = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype)
    shape = [1] * x.dim()
    shape[dim] = x.shape[dim]
    return total / n.view(shape)


def causal_prefix_attention(scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Prefix-softmax cross-attention: query row p attends kv columns j <= t.

    The causal counterpart of the (unmasked) cross-attention used by the
    perceiver-style memory summaries. Instead of one summary per window (which
    reads future tokens), output position ``t`` gets the same latent queries
    softmaxed over the PREFIX ``j <= t`` only::

        out[t, p] = sum_{j<=t} softmax_{j<=t}(scores[p, j]) @ v[j]

    Implemented as an explicitly masked softmax over the key axis, chunked
    over output positions so the (B, H, P, Tc, T) probability tensor stays
    around 2**26 elements (~256 MB fp32) — under the ~1 GB chunking budget
    even at the research shape (B=8, H=6, P=16, T=2048, Tc~85).

    CAUSALITY / CORRECTNESS NOTES (both learned the hard way):
      * Masked entries are filled with -inf BEFORE the softmax, so future
        tokens get weight exactly 0 — every output is a bitwise function of
        positions <= t (a plain allclose causality check passes at 0.0, not
        just under a tolerance).
      * A "cheap" prefix-softmax via cumsum of exp(s - running_max) is WRONG:
        the online-softmax rescaling does not telescope across a plain cumsum
        when the running max increases mid-sequence. And normalising by a
        GLOBAL max, while exact in real arithmetic, does not cancel in
        floating point and leaks future tokens at ~1e-8. Don't reintroduce
        either; the masked form is the reference.
      * The scores are floated to fp32 BEFORE the masked softmax, so the
        softmax and the einsum against ``vf = v.float()`` run in a single
        dtype (like the Sinkhorn helpers). Without the cast a pure-bf16 model
        keeps bf16 ``probs`` next to fp32 ``vf`` and the einsum raises
        ``RuntimeError: expected scalar type Float but found BFloat16``
        (pinned by the bf16 causal regression test); fp32 also keeps the
        prefix softmax stable when bf16 scores carry large magnitudes.

    Args:
        scores: (B, H, P, T) — ALREADY-scaled query-to-kv scores (P latent
            queries, T kv positions).
        v:      (B, H, T, hd) — kv values.
    Returns:
        (B, H, T, P, hd).
    """
    bsz, heads, points, seq = scores.shape
    head_dim = v.shape[-1]
    vf = v.float()
    # Float the scores once up front so probs matches vf's dtype (see the
    # bf16 note above); every chunk's masked_fill/softmax/einsum then runs
    # uniformly in fp32.
    scores = scores.float()
    # Output positions are chunked; the key axis is always the full prefix.
    step = max(1, min(seq, (2**26) // max(1, bsz * heads * points * seq)))
    chunks: list[torch.Tensor] = []
    device = scores.device
    cols = torch.arange(seq, device=device)
    for start in range(0, seq, step):
        end = min(start + step, seq)
        rows = torch.arange(start, end, device=device)
        # mask[i, j] = key j is visible to output row i (j <= i).
        mask = cols.unsqueeze(0) <= rows.unsqueeze(1)  # (Tc, T)
        # (B, H, P, Tc, T): broadcast the key scores across output rows, then
        # mask; softmax over the key axis is the prefix softmax per row.
        s = scores.unsqueeze(3).expand(bsz, heads, points, end - start, seq)
        probs = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
        probs = probs.transpose(2, 3)  # (B, H, Tc, P, T)
        chunks.append(torch.einsum("bhtpj,bhjd->bhtpd", probs, vf))
    out = torch.cat(chunks, dim=2) if len(chunks) > 1 else chunks[0]
    return out.to(v.dtype)


def causal_last_tokens(x: torch.Tensor, n: int) -> torch.Tensor:
    """(B, T, D) -> (B, T, n, D): the ``n`` tokens ENDING at each position.

    Causal counterpart of the legacy hierarchical short-memory write
    ``x[:, -n:]`` (the last n tokens of the window): at position t the legacy
    slice includes tokens after t, so the causal form takes the last n tokens
    of the PREFIX ending at t (left zero-padded exactly like the legacy path
    pads a too-short window).
    """
    padded = F.pad(x, (0, 0, n - 1, 0))  # (B, T + n - 1, D)
    # unfold appends the window dim last: (B, T, D, n) -> permute to (B, T, n, D)
    return padded.unfold(dimension=1, size=n, step=1).permute(0, 1, 3, 2).contiguous()


class SDPCrossAttention(nn.Module):
    """Compile-clean cross-attention via ``F.scaled_dot_product_attention``.

    Replaces ``nn.MultiheadAttention`` for the perceiver compression step in
    ``summary_memory`` and ``bloom_memory``. ``nn.MultiheadAttention``'s internal
    ``.view`` / ``.contiguous`` reshapes and projection layout graph-break under
    ``torch.compile`` (mode="reduce-overhead"), which drops the compiled region
    back to eager; eager then materialises the full ``(B, P, T)`` attention score
    matrix per layer and OOMs at batch 8 / seq 8192 (16 GiB allocation). SDPA is
    the fused flash / mem-efficient path and is compile-clean, so fixing the
    graph break fixes the OOM in the same stroke. See ``NEXT_IDEAS.md`` section 4.

    Functionally a drop-in for the cross-attention it replaces: Q comes from
    ``q_input`` (the perceiver latents), K=V come from ``kv_input`` (the full
    token window). No causal mask — this is cross-attention *into* the window,
    and the window's own causality is handled by the local self-attention layer.

    Param count is identical to the ``nn.MultiheadAttention`` it replaces
    (4*D*D + 4*D), so it does not perturb the param-matched bake-off. Default
    ``nn.Linear`` init matches PyTorch's MHA defaults (the closest behavioural
    match); ``out`` is tagged ``_is_residual_out`` so the depth-scaled init
    scheme (``init_scheme="scaled"`` / orthogonal) scales it like the other
    residual-output projections.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        bias = bool(getattr(config, "use_bias", True))
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=bias)
        self.out_proj._is_residual_out = True  # tagged for depth-scaled init

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, q_input: torch.Tensor, kv_input: torch.Tensor) -> torch.Tensor:
        # Project, split into heads, attend. Mirrors CausalSelfAttention._split /
        # _forward_sdpa (base.py) but with separate Q and K/V projections because
        # Q (latents) and KV (window) are different tensors. No attn_mask: this is
        # cross-attention, so SDPA never materialises a (P, T) score tensor.
        q = self._split(self.q_proj(q_input))
        k = self._split(self.k_proj(kv_input))
        v = self._split(self.v_proj(kv_input))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(q_input.shape[0], q_input.shape[1], self.num_heads * self.head_dim)
        return self.out_proj(out)

    def causal_forward(self, latents: torch.Tensor, kv_input: torch.Tensor) -> torch.Tensor:
        """Causal-memory path (config.causal_memory=True). DO NOT use otherwise.

        Same parameters and projections as ``forward`` (no new params), but
        every output position ``t`` gets the SAME latent queries attending to
        the PREFIX ``kv_input[:, :t+1]`` only, so the summary written into the
        memory state at t is a function of tokens <= t. Scaling matches
        ``forward``'s SDPA (1/sqrt(head_dim)).

        Args:
            latents:  (B, P, D) or (1, P, D) — learned perceiver queries.
            kv_input: (B, T, D) — the token window.
        Returns:
            (B, T, P, D): the per-position summaries that replace the single
            (B, P, D) whole-window summary in the causal write path.
        """
        bsz, seq_len, _ = kv_input.shape
        q = self._split(self.q_proj(latents))  # (1|B, H, P, hd)
        if q.shape[0] != bsz:
            q = q.expand(bsz, -1, -1, -1)
        k = self._split(self.k_proj(kv_input))  # (B, H, T, hd)
        v = self._split(self.v_proj(kv_input))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, H, P, T)
        ctx = causal_prefix_attention(scores, v)  # (B, H, T, P, hd)
        ctx = ctx.permute(0, 2, 3, 1, 4).reshape(bsz, seq_len, -1, self.num_heads * self.head_dim)  # (B, T, P, D)
        return self.out_proj(ctx)


class MemoryRead(nn.Module):
    def __init__(self, config: ModelConfig, flow: nn.Module | None = None) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.kernel_bias = config.memory_kernel_bias
        self.q = nn.Linear(config.d_model, config.d_model)
        self.kv = nn.Linear(config.d_model, config.d_model * 2)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.flow = flow
        if self.kernel_bias not in {"none", "positional", "rbf"}:
            raise ValueError("memory_kernel_bias must be none, positional, or rbf")
        self.slot_bias = nn.Parameter(torch.zeros(config.memory_slots + config.short_memory_slots))
        self.rbf_scale = nn.Parameter(torch.tensor(1.0))
        # Sweep 7 (B2): log-sum-exp energy read. For small beta this behaves like
        # a mean read; as beta grows it sharpens toward max-energy retrieval.
        self.energy_read = getattr(config, "energy_read", False)
        if self.energy_read:
            beta_init = float(getattr(config, "energy_beta_init", 1.0))
            self.energy_log_beta = nn.Parameter(torch.tensor(math.log(max(beta_init, 1e-6))))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _bias(self, q_len: int, m_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.kernel_bias == "none":
            return None
        if self.kernel_bias == "positional":
            return self.slot_bias[:m_len].to(device=device, dtype=dtype).view(1, 1, 1, m_len)
        q_pos = torch.linspace(0, 1, q_len, device=device, dtype=dtype).view(q_len, 1)
        m_pos = torch.linspace(0, 1, m_len, device=device, dtype=dtype).view(1, m_len)
        scale = self.rbf_scale.abs().to(dtype=dtype) + 1e-6
        return (-(q_pos - m_pos).pow(2) * scale).view(1, 1, q_len, m_len)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if memory.dim() == 4:
            # causal_memory=True: memory is a per-position state (B, T, S, D);
            # the read at token t consumes only the state at t.
            return self._forward_causal(x, memory)
        q = self._split(self.q(x))
        k, v = self.kv(memory).chunk(2, dim=-1)
        k, v = self._split(k), self._split(v)
        if self.flow is not None:
            q = self.flow(q)
            k = self.flow(k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        bias = self._bias(x.shape[1], memory.shape[1], x.device, scores.dtype)
        if bias is not None:
            scores = scores + bias
        if self.energy_read:
            beta = self.energy_log_beta.exp().clamp_min(1e-6).to(device=scores.device)
            scores_f = scores.float()
            v_f = v.float()
            log_partition = torch.logsumexp(beta.float() * scores_f, dim=-1).unsqueeze(-1)
            out = (
                torch.logsumexp(beta.float() * (scores_f.unsqueeze(-1) + v_f.unsqueeze(2)), dim=-2)
                - log_partition
            ) / beta.float()
            out = out.to(dtype=v.dtype)
        else:
            attn = torch.softmax(scores, dim=-1)
            out = attn @ v
        out = out.transpose(1, 2).contiguous().view(x.shape)
        return self.out(out)

    def _bias_causal(
        self, q_len: int, m_len: int, device: torch.Device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Kernel bias for the per-position read, broadcastable to (B, H, T, 1, S).

        Same construction as ``_bias`` but with an extra position axis: each
        token t is its own query row, so the rbf bias uses t's own normalised
        position rather than a shared (q_len, m_len) grid.
        """
        if self.kernel_bias == "none":
            return None
        if self.kernel_bias == "positional":
            return self.slot_bias[:m_len].to(device=device, dtype=dtype).view(1, 1, 1, 1, m_len)
        # NOTE (rbf grid is LENGTH-conditional, not token-value leaky): q_pos
        # normalises each token's position by the CURRENT sequence length
        # (linspace(0, 1, q_len)), exactly like the legacy read's grid. Token
        # t's bias therefore depends on T (position t/(T-1)), so running the
        # model on a truncated window rescales every query position and the
        # strict prefix-truncation invariance test sits at ~1.5e-5 — just
        # above the 1e-5 atol the kernel_bias="none" variants hold (the
        # truncation test covers rbf with an annotated relaxed tolerance).
        # This is NOT a causality leak: no future token VALUES enter the bias
        # (the exact-0 last-token perturbation test passes unchanged with
        # rbf), and the legacy read has the same T-dependence, so flag-on and
        # flag-off behave consistently.
        q_pos = torch.linspace(0, 1, q_len, device=device, dtype=dtype).view(q_len, 1)
        m_pos = torch.linspace(0, 1, m_len, device=device, dtype=dtype).view(1, m_len)
        scale = self.rbf_scale.abs().to(dtype=dtype) + 1e-6
        return (-(q_pos - m_pos).pow(2) * scale).view(1, 1, q_len, 1, m_len)

    def _forward_causal(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Per-position read for causal_memory=True. Same parameters as the
        legacy read; the token at position t attends over the memory state at
        t only (a (T, 1, S) attention per batch-head), so the read cannot see
        a memory state built from tokens after t."""
        bsz, seq_len, slots, _ = memory.shape
        q = self._split(self.q(x))  # (B, H, T, hd)
        k, v = self.kv(memory).chunk(2, dim=-1)  # (B, T, S, D) each
        k = k.reshape(bsz, seq_len, slots, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz, seq_len, slots, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        # (B, H, T, S, hd); the flow's MLP/time-embedding ops are last-dim ops
        # and work unchanged on the extra position axis.
        if self.flow is not None:
            q = self.flow(q)
            k = self.flow(k)
        scores = q.unsqueeze(-2) @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, H, T, 1, S)
        bias = self._bias_causal(seq_len, slots, x.device, scores.dtype)
        if bias is not None:
            scores = scores + bias
        if self.energy_read:
            beta = self.energy_log_beta.exp().clamp_min(1e-6).to(device=scores.device)
            scores_f = scores.float()
            v_f = v.float()
            log_partition = torch.logsumexp(beta.float() * scores_f, dim=-1).unsqueeze(-1)
            # scores_f.unsqueeze(-1): (B,H,T,1,S,1); v_f.unsqueeze(3): (B,H,T,1,S,hd);
            # logsumexp over the S axis (dim=-2) -> (B, H, T, 1, 1, hd).
            out = (
                torch.logsumexp(beta.float() * (scores_f.unsqueeze(-1) + v_f.unsqueeze(3)), dim=-2)
                - log_partition
            ) / beta.float()
            out = out.to(dtype=v.dtype).squeeze(3).squeeze(-2)  # (B, H, T, hd)
        else:
            attn = torch.softmax(scores, dim=-1)  # (B, H, T, 1, S)
            out = (attn @ v).squeeze(-2)  # (B, H, T, hd)
        # Merge heads back into the feature dim (same tail as the legacy read).
        out = out.transpose(1, 2).contiguous().view(x.shape)
        return self.out(out)


class MemoryMixinBlock(nn.Module):
    def initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def update_memory(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# Attribute names whose submodules switched from nn.MultiheadAttention to
# SDPCrossAttention. Each held an in_proj_weight / in_proj_bias / out_proj pair
# (nn.MultiheadAttention's storage layout); SDPCrossAttention uses four separate
# q_proj / k_proj / v_proj / out_proj Linears. Used by remap_legacy_mha_state_dict.
_LEGACY_MHA_ATTRS = ("perceiver", "summary_attn")


def remap_legacy_mha_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    bias: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Rewrite a pre-S14 summary/bloom checkpoint's ``nn.MultiheadAttention``
    parameters into the new ``SDPCrossAttention`` projection layout.

    ``SDPCrossAttention`` (see NEXT_IDEAS.md section 4) replaced the
    ``nn.MultiheadAttention`` perceiver cross-attention in ``summary_memory``
    (``perceiver.*``) and ``bloom_memory`` (``summary_attn.*``) because MHA
    graph-breaks under ``torch.compile`` and OOMs at long context. An old
    checkpoint stores, per block, four MHA tensors::

        blocks.{i}.{perceiver,summary_attn}.in_proj_weight  (3*D, D)
        blocks.{i}.{perceiver,summary_attn}.in_proj_bias    (3*D,)
        blocks.{i}.{perceiver,summary_attn}.out_proj.weight (D, D)
        blocks.{i}.{perceiver,summary_attn}.out_proj.bias   (D,)

    where ``in_proj_weight`` is the row-stacked ``[Wq; Wk; Wv]`` (the standard
    ``nn.MultiheadAttention`` layout) and ``in_proj_bias`` is ``[bq; bk; bv]``.
    This splits them into the three separate projection matrices /
    biases that ``SDPCrossAttention`` holds (``q_proj`` / ``k_proj`` / ``v_proj``),
    copies ``out_proj`` through, and drops the legacy keys so ``load_state_dict``
    (strict) succeeds.

    ``nn.MultiheadAttention`` *always* has bias, but ``SDPCrossAttention``
    respects ``config.use_bias``: a checkpoint trained under ``use_bias=False``
    (e.g. sweep13) still carries MHA bias tensors that the new bias-free module
    does not expect. Pass ``bias=False`` (from ``config.use_bias``) to drop the
    remapped bias keys so the strict load matches. Default ``None`` keeps biases
    (correct for the ``use_bias=True`` arms).

    Param-count preserving when ``bias`` matches the target: the new four-Linears
    hold exactly ``4*D*D + 4*D`` params with bias (identical to MHA), or
    ``4*D*D`` without.

    Idempotent: a state_dict already in the new format (no ``in_proj_weight``
    under any of ``_LEGACY_MHA_ATTRS``) passes through untouched. No-op for
    non-summary / non-bloom state_dicts.
    """
    attr_alt = "|".join(_LEGACY_MHA_ATTRS)
    in_proj_w_re = re.compile(rf"^(.*)\.({attr_alt})\.in_proj_weight$")
    in_proj_b_re = re.compile(rf"^(.*)\.({attr_alt})\.in_proj_bias$")
    out_proj_w_re = re.compile(rf"^(.*)\.({attr_alt})\.out_proj\.weight$")
    out_proj_b_re = re.compile(rf"^(.*)\.({attr_alt})\.out_proj\.bias$")

    # Index legacy entries by (prefix, attr) so we can emit the 4 new keys in one
    # place per group. A single pass collects; a second pass mutates.
    groups: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        for regex, tag in (
            (in_proj_w_re, "in_proj_weight"),
            (in_proj_b_re, "in_proj_bias"),
            (out_proj_w_re, "out_proj.weight"),
            (out_proj_b_re, "out_proj.bias"),
        ):
            m = regex.match(key)
            if m:
                groups.setdefault((m.group(1), m.group(2)), {})[tag] = tensor
                break

    if not groups:
        return state_dict  # already new-format (or not a summary/bloom checkpoint)

    keep_bias = bias if bias is not None else True
    for (prefix, attr), parts in groups.items():
        base = f"{prefix}.{attr}"
        if "in_proj_weight" in parts:
            wq, wk, wv = parts["in_proj_weight"].tensor_split(3, dim=0)
            state_dict[f"{base}.q_proj.weight"] = wq
            state_dict[f"{base}.k_proj.weight"] = wk
            state_dict[f"{base}.v_proj.weight"] = wv
            del state_dict[f"{base}.in_proj_weight"]
        if keep_bias and "in_proj_bias" in parts:
            bq, bk, bv = parts["in_proj_bias"].tensor_split(3, dim=0)
            state_dict[f"{base}.q_proj.bias"] = bq
            state_dict[f"{base}.k_proj.bias"] = bk
            state_dict[f"{base}.v_proj.bias"] = bv
            del state_dict[f"{base}.in_proj_bias"]
        elif not keep_bias and "in_proj_bias" in parts:
            # Target module is bias-free; drop the legacy bias so it isn't an
            # unexpected key in the strict load.
            del state_dict[f"{base}.in_proj_bias"]
        # out_proj -> out_proj: weight keeps its key in both layouts. Bias keeps
        # its key only when the target has bias; drop it otherwise.
        if "out_proj.weight" in parts:
            state_dict[f"{base}.out_proj.weight"] = parts["out_proj.weight"]
        if keep_bias and "out_proj.bias" in parts:
            state_dict[f"{base}.out_proj.bias"] = parts["out_proj.bias"]
        elif not keep_bias and f"{base}.out_proj.bias" in state_dict:
            del state_dict[f"{base}.out_proj.bias"]

    return state_dict
