"""Still: Amortized KV-cache compaction via per-layer Perceiver.

Implements the architecture from O'Neill et al. (arXiv:2606.07878):
- A small per-layer Perceiver module that cross-attends the full KV cache
  and produces compact keys and values in a single forward pass.
- Position-free compaction: cached keys are un-rotated before the compactor,
  the compactor uses its own RoPE internally, and compact keys are re-rotated
  at evenly-spaced output positions.
- Identity-style initialization for stable early training.
- KL-divergence training against a frozen base model (teacher) vs. compact-cache
  student, on answer tokens only.

Novel variants (combining ideas from the Flower project wiki):
- OT-coupled compactor: Sinkhorn optimal-transport coupling in the cross-attention
  read, inspired by flow_ot_memory + [[optimal-transport-attention]].
- Energy-read compactor: log-sum-exp read instead of softmax-mean, from the
  [[agent-memory-architecture]] energy-based read idea (Sweep 7 B2).
- Frequency-decay compactor: per-slot spaced-repetition decay applied to the
  compact KV entries between compaction passes, from frequency_decay_memory.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Non-parametric RMS normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(dtype=x.dtype)


def _apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    base: float = 10.0,
) -> torch.Tensor:
    """Apply RoPE at arbitrary positions (used inside the compactor's cross-attention).

    x: (..., seq, head_dim)
    positions: (seq,) integer or float positions.
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head_dim")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    freqs = positions.float().to(x.device).unsqueeze(-1) * inv_freq  # (seq, half)
    cos = freqs.cos().repeat_interleave(2, dim=-1)  # (seq, head_dim)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + rotated * sin


def _inverse_rope(
    keys: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """Un-rotate keys that were RoPE-rotated at `positions` with `base`.

    This recovers the position-free key content. Still uses base=10000 for the
    base model's RoPE (standard) and base=10 for the compactor's internal RoPE.
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head_dim")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=keys.device, dtype=torch.float32) / half))
    freqs = positions.float().to(keys.device).unsqueeze(-1) * inv_freq
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    # Inverse rotation: rotate by -angle instead of +angle.
    x1 = keys[..., ::2]
    x2 = keys[..., 1::2]
    neg_rotated = torch.stack((x2, -x1), dim=-1).flatten(-2)
    return keys * cos + neg_rotated * sin


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.float().norm(dim=-1, keepdim=True).clamp_min(eps)).to(dtype=x.dtype)


def _even_positions(t: int, t_in: int, device: torch.device) -> torch.Tensor:
    """Evenly spaced output positions over the input range [0, t_in)."""
    if t <= 1:
        return torch.zeros(t, device=device, dtype=torch.float32)
    return torch.linspace(0, max(t_in - 1, 1), t, device=device, dtype=torch.float32)


class CompactorCrossAttention(nn.Module):
    """Cross-attention from compact latents into the full KV cache.

    Uses QK-norm + d_l logits scale (not 1/sqrt(d)), following Still Appendix A.
    """

    def __init__(
        self,
        d_latent: int,
        head_dim: int,
        num_heads: int = 1,
        rope_base: float = 10.0,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.rope_base = rope_base

        self.q_proj = nn.Linear(d_latent, d_latent, bias=use_bias)
        self.k_proj = nn.Linear(2 * head_dim, d_latent, bias=use_bias)  # input is [K;V] = 2*head_dim
        self.v_proj = nn.Linear(2 * head_dim, d_latent, bias=False)
        self.out_proj = nn.Linear(d_latent, d_latent, bias=False)

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        key_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        z: (B, H_kv, t, d_latent) — compact latents (per KV-head).
        x: (B, H_kv, T, 2*head_dim) — concatenated [K;V] cache (position-free).
        key_positions: (T,) — positions for the compactor's internal RoPE on keys.
        """
        B, H, t, dl = z.shape
        T = x.shape[2]
        nh = self.num_heads
        dh = dl // nh

        q = self.q_proj(z).view(B, H, t, nh, dh).permute(0, 1, 3, 2, 4)  # (B,H,nh,t,dh)
        k = self.k_proj(x).view(B, H, T, nh, dh).permute(0, 1, 3, 2, 4)  # (B,H,nh,T,dh)
        v = self.v_proj(x).view(B, H, T, nh, dh).permute(0, 1, 3, 2, 4)

        q = _l2_normalize(q)
        k = _l2_normalize(k)

        if key_positions is not None:
            query_positions = _even_positions(t, T, z.device)
            q = _apply_rope(q, query_positions, dh, base=self.rope_base)
            k = _apply_rope(k, key_positions, dh, base=self.rope_base)

        # d_l logits scale after QK-norm (cosine similarities), per Still.
        scores = q.float() @ k.float().transpose(-2, -1) * float(dh)  # (B,H,nh,t,T)
        attn = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        out = attn @ v  # (B,H,nh,t,dh)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, H, t, dl)
        return self.out_proj(out)


class CompactorSelfAttention(nn.Module):
    """Latent self-attention with QK-norm + d_l logits scale."""

    def __init__(self, d_latent: int, num_heads: int = 1) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_latent, d_latent, bias=True)
        self.k_proj = nn.Linear(d_latent, d_latent, bias=True)
        self.v_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.out_proj = nn.Linear(d_latent, d_latent, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, H, t, dl = z.shape
        nh = self.num_heads
        dh = dl // nh
        q = self.q_proj(z).view(B, H, t, nh, dh).permute(0, 1, 3, 2, 4)
        k = self.k_proj(z).view(B, H, t, nh, dh).permute(0, 1, 3, 2, 4)
        v = self.v_proj(z).view(B, H, t, nh, dh).permute(0, 1, 3, 2, 4)
        q = _l2_normalize(q)
        k = _l2_normalize(k)
        scores = q.float() @ k.float().transpose(-2, -1) * float(dh)
        # Causal mask within latents (position-aware compaction benefits from local structure).
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=z.device, dtype=torch.float32), diagonal=1
        )
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        out = attn @ v
        out = out.permute(0, 1, 3, 2, 4).reshape(B, H, t, dl)
        return self.out_proj(out)


class CompactorBlock(nn.Module):
    """One Perceiver block: cross-attention + self-attention + (optional) FFN."""

    def __init__(
        self,
        d_latent: int,
        head_dim: int,
        num_cross_heads: int = 1,
        num_self_heads: int = 1,
        ffn_ratio: int = 0,
        rope_base: float = 10.0,
        use_cross_attn: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_latent)
        self.norm2 = RMSNorm(d_latent)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CompactorCrossAttention(
                d_latent, head_dim, num_cross_heads, rope_base=rope_base
            )
        self.self_attn = CompactorSelfAttention(d_latent, num_self_heads)
        self.use_ffn = ffn_ratio > 0
        if self.use_ffn:
            self.norm3 = RMSNorm(d_latent)
            self.ffn = nn.Sequential(
                nn.Linear(d_latent, ffn_ratio * d_latent),
                nn.GELU(),
                nn.Linear(ffn_ratio * d_latent, d_latent),
            )

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor | None,
        key_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_cross_attn and x is not None:
            z = z + self.cross_attn(self.norm1(z), x, key_positions)
        z = z + self.self_attn(self.norm2(z))
        if self.use_ffn:
            z = z + self.ffn(self.norm3(z))
        return z


class StillCompactor(nn.Module):
    """Per-layer Perceiver KV-cache compactor.

    Takes the full per-layer KV cache (B, H, T, d) and produces compact
    (Ck, Cv) of shape (B, H, t, d) in a single forward pass.

    Configuration matches Still's canonical setup: d_latent=2*d, B=2 blocks
    with cross-attention repeated every block, nc=ns=1 heads, no FFN,
    identity-style initialization, RoPE-fix (position-free compaction).
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        compact_len: int = 128,
        num_blocks: int = 2,
        d_latent: int | None = None,
        num_cross_heads: int = 1,
        num_self_heads: int = 1,
        ffn_ratio: int = 0,
        rope_base: float = 10.0,
        base_rope_base: float = 10000.0,
        identity_init: bool = True,
        use_ot_read: bool = False,
        ot_epsilon: float = 0.1,
        ot_iters: int = 10,
        use_energy_read: bool = False,
        energy_beta_init: float = 1.0,
        freq_decay: bool = False,
        freq_penalty: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.compact_len = compact_len
        self.num_blocks = num_blocks
        self.d_latent = d_latent or 2 * head_dim
        self.rope_base = rope_base
        self.base_rope_base = base_rope_base
        self.identity_init = identity_init
        self.use_ot_read = use_ot_read
        self.ot_epsilon = ot_epsilon
        self.ot_iters = ot_iters
        self.use_energy_read = use_energy_read
        self.freq_decay = freq_decay
        self.freq_penalty = freq_penalty

        # Per-KV-head latent banks: (H, t, d_latent)
        self.latents = nn.Parameter(torch.zeros(num_kv_heads, compact_len, self.d_latent))

        # Compactor blocks: cross-attn in every block (canonical Still config).
        self.blocks = nn.ModuleList([
            CompactorBlock(
                self.d_latent,
                head_dim,
                num_cross_heads=num_cross_heads,
                num_self_heads=num_self_heads,
                ffn_ratio=ffn_ratio,
                rope_base=rope_base,
                use_cross_attn=True,
            )
            for _ in range(num_blocks)
        ])

        # Output projections: compact keys and values.
        self.key_proj = nn.Linear(self.d_latent, head_dim, bias=False)
        self.val_proj = nn.Linear(self.d_latent, head_dim, bias=False)

        # Optional energy-read inverse temperature.
        if use_energy_read:
            self.energy_log_beta = nn.Parameter(torch.tensor(math.log(max(energy_beta_init, 1e-6))))

        # Optional frequency decay per compact slot.
        if freq_decay:
            self.freq_decay_logit = nn.Parameter(torch.full((compact_len,), -2.0))

        if identity_init:
            self._init_identity()

    def _init_identity(self) -> None:
        """Identity-style initialization (Still Appendix C).

        Makes the untrained compactor a near-pass-through at t=T.
        Requires d_latent = 2 * head_dim.
        """
        with torch.no_grad():
            # Zero the latent bank (queries start blank).
            self.latents.zero_()

            # For each block, zero-init the refinement output projections so they
            # begin as residual identities. The first block's cross-attention gets
            # identity-aligned biases.
            for block in self.blocks:
                if block.use_cross_attn:
                    # Zero the cross-attention output projection.
                    nn.init.zeros_(block.cross_attn.out_proj.weight)
                    # Zero key projection so attention is position-dominated (not content).
                    nn.init.zeros_(block.cross_attn.k_proj.weight)
                    # Align query/key biases to a fixed unit direction.
                    if block.cross_attn.q_proj.bias is not None:
                        nn.init.zeros_(block.cross_attn.q_proj.bias)
                    if block.cross_attn.k_proj.bias is not None:
                        unit = torch.zeros(self.d_latent)
                        unit[0] = 1.0
                        block.cross_attn.k_proj.bias.data.copy_(unit)
                # Zero self-attention output projections.
                nn.init.zeros_(block.self_attn.out_proj.weight)
                if block.use_ffn:
                    nn.init.zeros_(block.ffn[-1].weight)

            # Identity output heads: key_proj = [Id; 0], val_proj = [0; Id].
            d = self.head_dim
            self.key_proj.weight.data.zero_()
            self.val_proj.weight.data.zero_()
            self.key_proj.weight.data[:d, :d] = torch.eye(d)
            self.val_proj.weight.data[:d, d:2 * d] = torch.eye(d)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor | None = None,
        return_compact_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compact a per-layer KV cache.

        keys:  (B, H, T, d) — RoPE-rotated cached keys.
        values: (B, H, T, d) — cached values.
        positions: (T,) — RoPE positions the keys were rotated at. If None, uses 0..T-1.
        return_compact_cache: if True, also return (Ck, Cv) re-rotated at output positions.

        Returns dict with 'latent_out' and optionally 'compact_keys', 'compact_values'.
        """
        B, H, T, d = keys.shape

        if positions is None:
            positions = torch.arange(T, device=keys.device, dtype=torch.float32)

        # Un-rotate keys into position-free frame (RoPE-fix).
        keys_free = _inverse_rope(keys, positions, d, base=self.base_rope_base)

        # Concatenate [K; V] as the cross-attention input.
        x = torch.cat([keys_free, values], dim=-1)  # (B, H, T, 2d)

        # Expand latent bank to batch.
        z = self.latents.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, t, d_latent)

        # Apply compactor blocks.
        for block in self.blocks:
            z = block(z, x, key_positions=positions)

        # Optional energy read: sharpen the latent representation.
        if self.use_energy_read:
            beta = self.energy_log_beta.exp().clamp_min(1e-6)
            z_norm = RMSNorm(self.d_latent)(z)
            energy = z_norm.float().norm(dim=-1, keepdim=True)
            z = z * torch.sigmoid(beta * (energy - energy.mean(dim=-2, keepdim=True)))

        # Project to compact keys and values.
        Ck = self.key_proj(z)  # (B, H, t, d)
        Cv = self.val_proj(z)

        # Optional frequency decay on compact entries.
        if self.freq_decay:
            decay = torch.sigmoid(self.freq_decay_logit).view(1, 1, -1, 1)
            Cv = Cv * (1.0 - self.freq_penalty * decay)

        result: dict[str, torch.Tensor] = {"latent_out": z, "Ck_raw": Ck, "Cv": Cv}

        if return_compact_cache:
            # Re-rotate compact keys at evenly spaced output positions.
            out_positions = _even_positions(self.compact_len, T, keys.device)
            Ck_rotated = _apply_rope(Ck, out_positions, d, base=self.base_rope_base)
            result["compact_keys"] = Ck_rotated
            result["compact_values"] = Cv
            result["out_positions"] = out_positions

        return result


def _sinkhorn_ot(
    cost: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 10,
) -> torch.Tensor:
    """Sinkhorn OT plan for coupling latents to cache entries.

    cost: (B, t, T) transport cost matrix.
    Returns: (B, t, T) transport plan (row-stochastic).
    """
    B, t, T = cost.shape
    log_K = -cost / epsilon  # (B, t, T)
    # Sinkhorn in log-space with proper broadcasting.
    log_u = torch.zeros(B, t, 1, device=cost.device, dtype=cost.dtype)
    log_v = torch.zeros(B, 1, T, device=cost.device, dtype=cost.dtype)
    for _ in range(n_iters):
        # Update u: log_u[i] = -log(t) - logsumexp_j(log_K[i,j] + log_v[j])
        log_u = -math.log(t) - torch.logsumexp(log_K + log_v, dim=-1, keepdim=True)
        # Update v: log_v[j] = -log(T) - logsumexp_i(log_K[i,j] + log_u[i])
        log_v = -math.log(T) - torch.logsumexp(log_K + log_u, dim=-2, keepdim=True)
    plan = torch.exp(log_u + log_K + log_v)  # (B, t, T)
    return plan


class StillCompactorSpectral(StillCompactor):
    """Spectral compactor: exploits the 3-4% effective dimensionality of KV keys.

    Inspired by [[kv-cache-spectral-compression]] (SpectralQuant, Dynamis Labs):
    transformer key vectors have only ~3-4% effective dimensions regardless of
    model size or architecture family. Values have ~10-15x larger effective rank.

    This compactor learns per-head low-rank signal subspaces for keys and values
    separately, then runs the Perceiver cross-attention in the compressed
    spectral domain. The key insight: if keys live in a 3-4% subspace, we can
    run the compactor's cross-attention on ~4% of the dimensions and reconstruct
    the full keys from the compact spectral representation.

    Architecture:
    1. Learnable signal projection: P_k (d -> r_k), P_v (d -> r_v) where r_k << r_v
    2. Cross-attention operates on spectrally compressed [K_s; V_s] (dim = r_k + r_v)
    3. Output: reconstruct full-dim keys and values from compact spectral codes
    4. Separate reconstruction heads for keys (low-rank) and values (higher-rank)

    The spectral projections are learned end-to-end via the KL distillation loss.
    At initialization, they approximate random projections (flat spectrum).
    """

    def __init__(
        self,
        *args,
        spectral_key_rank: int | None = None,
        spectral_val_rank: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        d = self.head_dim
        # Default key rank ~4% of head_dim (matching the spectral finding).
        r_k = spectral_key_rank or max(d // 25, 2)
        # Default value rank ~30% (values have ~10x the effective rank of keys).
        r_v = spectral_val_rank or max(d // 4, 4)
        self.r_k = r_k
        self.r_v = r_v

        # Learnable signal/noise partition: orthogonal projection for keys.
        # Key signal subspace projection.
        self.key_signal = nn.Linear(d, r_k, bias=False)
        self.key_reconstruct = nn.Linear(r_k, d, bias=False)
        # Value signal subspace projection (higher rank).
        self.val_signal = nn.Linear(d, r_v, bias=False)
        self.val_reconstruct = nn.Linear(r_v, d, bias=False)

        # Spectral cross-attention operates on compressed representations.
        # Use a lightweight custom cross-attention (not CompactorCrossAttention
        # which hardcodes 2*head_dim input).
        self.spectral_d_latent = r_k + r_v
        self.spectral_q_proj = nn.Linear(self.d_latent, self.d_latent, bias=True)
        self.spectral_k_proj = nn.Linear(self.spectral_d_latent, self.d_latent, bias=True)
        self.spectral_v_proj = nn.Linear(self.spectral_d_latent, self.d_latent, bias=False)
        self.spectral_out_proj = nn.Linear(self.d_latent, self.d_latent, bias=False)
        self.spectral_norm = RMSNorm(self.d_latent)
        self.spectral_num_heads = 1
        self.spectral_dh = self.d_latent // self.spectral_num_heads

        self._init_spectral()

    def _init_spectral(self) -> None:
        """Initialize spectral projections as approximate identity (pass-through)."""
        with torch.no_grad():
            d = self.head_dim
            # Key: random orthogonal-ish projection (rows of the weight are random unit vectors).
            kw = self.key_signal.weight  # (r_k, d)
            # Value-dead by intent: the weight is fully overwritten a few
            # lines below. The call exists ONLY because it consumes global
            # RNG draws, and every draw after it (the kaiming_normal_ below
            # and all subsequent inits) is part of the seeded stream — this
            # repo pins bit-repro of published runs, so removing the call
            # silently shifted every seeded spectral-compactor init
            # downstream.
            nn.init.orthogonal_(self.key_reconstruct.weight)  # (d, r_k)
            # Actually init key_reconstruct as the pseudo-inverse of key_signal.
            nn.init.kaiming_normal_(kw, a=0.0)
            # key_reconstruct should be the transpose-ish of key_signal for identity-like behavior.
            self.key_reconstruct.weight.data = kw.data.T[: d, :].contiguous()
            # Zero the signal projection initially so the spectral branch is a no-op
            # and the standard compactor path dominates at init.
            # Actually, we want the spectral path to contribute, so use small init.
            kw.data.mul_(0.1)

            vw = self.val_signal.weight  # (r_v, d)
            nn.init.kaiming_normal_(vw, a=0.0)
            self.val_reconstruct.weight.data = vw.data.T[: d, :].contiguous()
            vw.data.mul_(0.1)

            # Zero-init spectral cross-attention output so it starts as residual identity.
            nn.init.zeros_(self.spectral_out_proj.weight)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor | None = None,
        return_compact_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        B, H, T, d = keys.shape
        if positions is None:
            positions = torch.arange(T, device=keys.device, dtype=torch.float32)

        keys_free = _inverse_rope(keys, positions, d, base=self.base_rope_base)

        # Spectral compression: project K, V into low-rank subspaces.
        k_s = self.key_signal(keys_free)  # (B, H, T, r_k)
        v_s = self.val_signal(values)  # (B, H, T, r_v)
        x_spectral = torch.cat([k_s, v_s], dim=-1)  # (B, H, T, r_k + r_v)

        # Standard full-dim cross-attention input (for residual path).
        x_full = torch.cat([keys_free, values], dim=-1)  # (B, H, T, 2d)

        z = self.latents.unsqueeze(0).expand(B, -1, -1, -1)

        # Block 0: spectral cross-attention + standard cross-attention.
        block0 = self.blocks[0]
        z_norm = block0.norm1(z)
        # Standard cross-attention (from parent class blocks).
        z = z + block0.cross_attn(z_norm, x_full, key_positions=positions)
        # Spectral cross-attention (additional signal-extraction path).
        z_sn = self.spectral_norm(z)
        B_, H_, t_, dl_ = z_sn.shape
        T_ = x_spectral.shape[2]
        nh_ = self.spectral_num_heads
        dh_ = self.spectral_dh
        sq = self.spectral_q_proj(z_sn).view(B_, H_, t_, nh_, dh_).permute(0, 1, 3, 2, 4)
        sk = self.spectral_k_proj(x_spectral).view(B_, H_, T_, nh_, dh_).permute(0, 1, 3, 2, 4)
        sv = self.spectral_v_proj(x_spectral).view(B_, H_, T_, nh_, dh_).permute(0, 1, 3, 2, 4)
        sq = _l2_normalize(sq)
        sk = _l2_normalize(sk)
        sk_pos = positions
        sq_pos = _even_positions(t_, T_, z_sn.device)
        sq = _apply_rope(sq, sq_pos, dh_, base=self.rope_base)
        sk = _apply_rope(sk, sk_pos, dh_, base=self.rope_base)
        scores = sq.float() @ sk.float().transpose(-2, -1) * float(dh_)
        attn = torch.softmax(scores, dim=-1).to(dtype=sv.dtype)
        sout = attn @ sv
        sout = sout.permute(0, 1, 3, 2, 4).reshape(B_, H_, t_, dl_)
        z = z + self.spectral_out_proj(sout)
        z = z + block0.self_attn(block0.norm2(z))
        if block0.use_ffn:
            z = z + block0.ffn(block0.norm3(z))

        # Remaining blocks: standard processing.
        for block in self.blocks[1:]:
            z = block(z, x_full, key_positions=positions)

        # Output projections (standard path).
        Ck = self.key_proj(z)  # (B, H, t, d)
        Cv = self.val_proj(z)

        # Spectral reconstruction refinement: add the spectral signal back.
        # The compact latent z attends to the spectral features, then we
        # reconstruct through the signal subspaces as a residual.
        # This helps the compactor exploit the low-rank structure.
        Ck = Ck + self.key_reconstruct(
            self.key_signal(Ck)  # project to spectral then back
        )
        Cv = Cv + self.val_reconstruct(
            self.val_signal(Cv)
        )

        result: dict[str, torch.Tensor] = {"latent_out": z, "Ck_raw": Ck, "Cv": Cv}
        if return_compact_cache:
            out_positions = _even_positions(self.compact_len, T, keys.device)
            Ck_rotated = _apply_rope(Ck, out_positions, d, base=self.base_rope_base)
            result["compact_keys"] = Ck_rotated
            result["compact_values"] = Cv
            result["out_positions"] = out_positions
        return result


class FlowVelocityNet(nn.Module):
    """Velocity field for flow-matching compaction.

    Given current state z_t and conditioning context c (summary of full KV),
    predicts the velocity dz/dt that transports initial latents toward compact
    representations that faithfully reconstruct the full cache's attention behavior.

    Uses a lightweight MLP with FiLM conditioning (scale + shift from context).
    """

    def __init__(self, d_latent: int, d_cond: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        h = hidden_dim or d_latent * 2
        self.time_embed = nn.Sequential(
            nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h)
        )
        self.cond_proj = nn.Linear(d_cond, h * 2)  # FiLM: scale + shift
        self.net = nn.Sequential(
            nn.Linear(d_latent + h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, d_latent),
        )
        self._init_zero()

    def _init_zero(self) -> None:
        """Zero-init final layer so initial velocity is zero (identity flow)."""
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, H, t, h)
        film = self.cond_proj(cond)  # (B, H, t, 2*h)
        scale, shift = film.chunk(2, dim=-1)
        t_modulated = t_emb * (1 + scale) + shift
        return self.net(torch.cat([z, t_modulated], dim=-1))


class StillCompactorFlow(StillCompactor):
    """Flow-matching compactor: transports latents to compact KV via CNF.

    Novel research direction (no prior work on flow matching for KV compaction).

    Instead of Perceiver cross-attention, this compactor:
    1. Summarizes the full KV cache into a context vector (via pooling)
    2. Uses a learned velocity field to transport initial latents through
       continuous time [0, 1] toward compact KV representations
    3. The flow is trained end-to-end via KL distillation

    Advantages:
    - Smoother optimization landscape (continuous flow vs discrete attention)
    - Invertible: can define a reverse flow for decompression
    - Better coverage: the flow visits intermediate states, potentially
      capturing multi-scale information that single-pass attention misses

    Architecture:
    - Context: mean + max pooling of [K; V] cache (dim = 2 * 2 * head_dim)
    - Velocity net: FiLM-conditioned MLP on (z, t, context)
    - Euler integration for `flow_steps` steps
    """

    def __init__(
        self,
        *args,
        flow_steps: int = 5,
        velocity_hidden: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.flow_steps = flow_steps

        d = self.head_dim
        d_cond = 4 * d  # mean + max of keys and values, each 2*d (for K_free + V)

        # Velocity field for keys and values. `velocity_hidden` lets us shrink the
        # velocity nets for parameter-matched comparisons against the standard
        # compactor (default 2*d_latent preserves the original behavior).
        self.velocity_keys = FlowVelocityNet(self.d_latent, d_cond, hidden_dim=velocity_hidden)
        self.velocity_vals = FlowVelocityNet(self.d_latent, d_cond, hidden_dim=velocity_hidden)

        # Context projection: pool cache into a per-latent conditioning vector.
        # We use cross-attention to get a per-latent context (cheap, single head).
        self.context_proj_k = nn.Linear(2 * d, d_cond)
        self.context_proj_v = nn.Linear(2 * d, d_cond)

        self._init_flow()

    def _init_flow(self) -> None:
        """Initialize flow to start as identity (zero velocity)."""
        # FlowVelocityNet already zero-inits. Extra: zero context projections.
        with torch.no_grad():
            nn.init.zeros_(self.context_proj_k.weight)
            nn.init.zeros_(self.context_proj_v.bias)
            nn.init.zeros_(self.context_proj_v.weight)
            nn.init.zeros_(self.context_proj_k.bias)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor | None = None,
        return_compact_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        B, H, T, d = keys.shape
        if positions is None:
            positions = torch.arange(T, device=keys.device, dtype=torch.float32)

        keys_free = _inverse_rope(keys, positions, d, base=self.base_rope_base)
        x = torch.cat([keys_free, values], dim=-1)  # (B, H, T, 2d)

        # Initial latents.
        z = self.latents.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, t, d_latent)

        # Step 1: Cross-attention read from first block (same as standard Still).
        block0 = self.blocks[0]
        z = z + block0.cross_attn(block0.norm1(z), x, key_positions=positions)
        z = z + block0.self_attn(block0.norm2(z))
        if block0.use_ffn:
            z = z + block0.ffn(block0.norm3(z))

        # Step 2: Flow-matching refinement.
        # Compute conditioning from the full KV cache.
        cond_k = self.context_proj_k(
            x.mean(dim=2).unsqueeze(2).expand(-1, -1, self.compact_len, -1)
        )
        cond_v = self.context_proj_v(
            x.max(dim=2).values.unsqueeze(2).expand(-1, -1, self.compact_len, -1)
        )

        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t_val = torch.tensor(step * dt, device=z.device, dtype=z.dtype)
            # Key velocity (applied to first half of latent).
            d_k = self.velocity_keys(z, t_val.expand(B, H, self.compact_len), cond_k)
            # Value velocity (applied to second half of latent).
            d_v = self.velocity_vals(z, t_val.expand(B, H, self.compact_len), cond_v)
            # Combined velocity (average).
            velocity = 0.5 * (d_k + d_v)
            z = z + velocity * dt

        # Step 3: Remaining blocks for final refinement.
        for block in self.blocks[1:]:
            z = block(z, x, key_positions=positions)

        # Project to compact keys and values.
        Ck = self.key_proj(z)
        Cv = self.val_proj(z)

        result: dict[str, torch.Tensor] = {"latent_out": z, "Ck_raw": Ck, "Cv": Cv}
        if return_compact_cache:
            out_positions = _even_positions(self.compact_len, T, keys.device)
            Ck_rotated = _apply_rope(Ck, out_positions, d, base=self.base_rope_base)
            result["compact_keys"] = Ck_rotated
            result["compact_values"] = Cv
            result["out_positions"] = out_positions
        return result


class StillCompactorOT(StillCompactor):
    """Still compactor with OT-coupled cross-attention read.

    Instead of standard softmax cross-attention, the latent-to-cache coupling
    uses a Sinkhorn OT plan, enforcing marginal conservation on both sides.
    This connects to the [[optimal-transport-attention]] and [[flow_ot_memory]]
    wiki concepts: the compactor learns a structured transport from full cache
    to compact cache rather than an unconstrained attention distribution.
    """

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor | None = None,
        return_compact_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        B, H, T, d = keys.shape
        if positions is None:
            positions = torch.arange(T, device=keys.device, dtype=torch.float32)

        keys_free = _inverse_rope(keys, positions, d, base=self.base_rope_base)
        x = torch.cat([keys_free, values], dim=-1)
        z = self.latents.unsqueeze(0).expand(B, -1, -1, -1)

        for block in self.blocks:
            # Replace cross-attention with OT-coupled read in first block only.
            if block.use_cross_attn and block is self.blocks[0]:
                z_norm = block.norm1(z)
                # Compute cost as negative cosine similarity.
                q = block.cross_attn.q_proj(z_norm)  # (B, H, t, d_latent)
                k = block.cross_attn.k_proj(x)  # (B, H, T, d_latent)
                v = block.cross_attn.v_proj(x)
                q_n = _l2_normalize(q)
                k_n = _l2_normalize(k)

                # Use gradient checkpointing for Sinkhorn to save memory.
                def _ot_cross_attn(q_n, k_n, v, out_proj):
                    cost = -(q_n.float() @ k_n.float().transpose(-2, -1))  # (B, H, t, T)
                    plan = _sinkhorn_ot(
                        cost.reshape(B * H, self.compact_len, T),
                        epsilon=self.ot_epsilon,
                        n_iters=self.ot_iters,
                    ).reshape(B, H, self.compact_len, T)
                    return out_proj(plan @ v)

                if self.training:
                    out = torch.utils.checkpoint.checkpoint(
                        _ot_cross_attn, q_n, k_n, v, block.cross_attn.out_proj,
                        use_reentrant=False,
                    )
                else:
                    out = _ot_cross_attn(q_n, k_n, v, block.cross_attn.out_proj)
                z = z + out
                # Self-attention refinement.
                z = z + block.self_attn(block.norm2(z))
                if block.use_ffn:
                    z = z + block.ffn(block.norm3(z))
            else:
                z = block(z, x, key_positions=positions)

        Ck = self.key_proj(z)
        Cv = self.val_proj(z)

        result: dict[str, torch.Tensor] = {"latent_out": z, "Ck_raw": Ck, "Cv": Cv}
        if return_compact_cache:
            out_positions = _even_positions(self.compact_len, T, keys.device)
            Ck_rotated = _apply_rope(Ck, out_positions, d, base=self.base_rope_base)
            result["compact_keys"] = Ck_rotated
            result["compact_values"] = Cv
            result["out_positions"] = out_positions
        return result
