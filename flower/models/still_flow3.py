"""Still flow-matching compactor with K/V split + AdaLN-Zero velocity fields.

Two improvements on top of `StillCompactorFlow` (in `still.py`):

1. `AdaLNVelocityNet` -- upgrades the FiLM-conditioned `FlowVelocityNet` with
   Adaptive Layer Norm + Zero initialisation (AdaLN-Zero), the documented
   scaling recipe for flow velocity networks (DiT, Stable Diffusion 3). The
   sinusoidal timestep is fused with the KV-cache conditioning through
   per-layer scale/shift modulation, and both the modulation layers and the
   final output layer are zero-initialised so the initial velocity is exactly
   zero (the flow starts as the identity map).

2. `StillCompactorFlowKV` -- splits the latent bank into a key half and a
   value half and transports each half with a dedicated velocity field. The
   key field is smaller (keys live in a ~rank-4 subspace, ~3-4% effective
   dimensions per the Flower wiki spectral analysis) and the value field is
   larger (values are higher-rank, d_eff ~ 40-55). Each field conditions on
   only its own cache statistics, so the key flow never pays attention to
   value statistics and vice versa.

Both classes keep the same `forward(keys, values, positions=None,
return_compact_cache=False) -> dict` contract as `StillCompactor`, so they
drop directly into the existing `StillLM` KL-distillation training pipeline.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# Reuse the helpers and base architecture from still.py. The symbols that are
# not directly referenced inside this module are re-exported here on purpose so
# downstream code has a single import location; they are tagged noqa: F401 to
# silence ruff's unused-import warning for intentional re-exports.
from flower.models.still import (
    CompactorBlock,  # noqa: F401  (re-exported)
    CompactorCrossAttention,  # noqa: F401  (re-exported)
    CompactorSelfAttention,  # noqa: F401  (re-exported)
    FlowVelocityNet,  # noqa: F401  (re-exported, predecessor of AdaLNVelocityNet)
    RMSNorm,  # noqa: F401  (re-exported)
    StillCompactor,
    _apply_rope,
    _even_positions,
    _inverse_rope,
    _l2_normalize,  # noqa: F401  (re-exported)
    _sinkhorn_ot,  # noqa: F401  (re-exported)
)


class SinusoidalTimestepEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding for a scalar timestep.

    Given a tensor `t` of arbitrary shape, returns a tensor of shape
    `(*t.shape, dim)` where the first `dim // 2` channels are `cos(omega_i * t)`
    and the last `dim // 2` are `sin(omega_i * t)`, with geometrically-spaced
    frequencies `omega_i = exp(-log(max_period) * i / (dim // 2))`.
    """

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / float(half)
        )  # (half,)
        # Broadcast product: (*t.shape, half)
        args = t.float().unsqueeze(-1) * freqs
        emb = torch.cat([args.cos(), args.sin()], dim=-1)  # (*t.shape, dim)
        if self.dim % 2 == 1:  # pad to exactly `dim` if it was odd
            emb = F.pad(emb, (0, 1))
        return emb


class AdaLNVelocityNet(nn.Module):
    """Flow velocity field with AdaLN-Zero time conditioning.

    Replaces the FiLM modulation in `FlowVelocityNet` with the
    Adaptive-LayerNorm-with-Zero-init recipe used in DiT / Stable Diffusion 3:

    * timestep `t` is encoded with a sinusoidal embedding and a small MLP;
    * at every MLP layer the hidden state is normalised and then modulated by
      a per-layer (gamma, beta) pair produced from the fused time+cond vector;
    * both the per-layer modulation linears and the final output projection
      are zero-initialised, so the velocity is exactly zero at the start of
      training and the flow begins as the identity map.

    Inputs / outputs follow the `FlowVelocityNet` calling convention:
        z:    (B, H, Tz, d_latent) -- current state of the latents.
        t:    (B, H, Tz)            -- flow timestep (same spatial shape as z).
        cond: (B, H, Tz, d_cond)    -- per-latent conditioning vector.
        -> velocity: (B, H, Tz, d_latent).
    """

    def __init__(
        self,
        d_latent: int,
        d_cond: int,
        num_layers: int = 3,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.d_cond = d_cond
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim or d_latent * 2

        # Time pathway: sinusoidal embedding -> 2-layer MLP -> d_cond.
        self.time_embed = SinusoidalTimestepEmbedding(self.d_cond)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_cond, self.d_cond),
            nn.SiLU(),
            nn.Linear(self.d_cond, self.d_cond),
        )

        # Conditioning pathway: project the KV-cache summary to d_cond so it can
        # be added to the time embedding.
        self.cond_proj = nn.Linear(self.d_cond, self.d_cond)

        # Input lift from d_latent into the hidden width.
        self.input_proj = nn.Linear(self.d_latent, self.hidden_dim)

        # Per-layer AdaLN modulation: d_cond -> (gamma, beta) of width hidden_dim.
        self.adaln_mods = nn.ModuleList(
            [nn.Linear(self.d_cond, 2 * self.hidden_dim) for _ in range(num_layers)]
        )
        # LayerNorm without learnable affine -- the affine role is played by
        # the AdaLN (gamma, beta).
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim, elementwise_affine=False) for _ in range(num_layers)]
        )
        # Main MLP path: hidden_dim -> hidden_dim per layer.
        self.linears = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(num_layers)]
        )

        # Final projection: hidden_dim -> d_latent (zero-init for identity flow).
        self.out_proj = nn.Linear(self.hidden_dim, self.d_latent)

        self._init_zero()

    def _init_zero(self) -> None:
        """Zero-init AdaLN modulations and final layer.

        With gamma = beta = 0 the AdaLN output is zero, and with a zero final
        projection the overall velocity is identically zero, so the flow starts
        as the identity map regardless of input.
        """
        for mod in self.adaln_mods:
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        # Time embedding: sinusoidal -> MLP -> (*, d_cond).
        t_emb = self.time_mlp(self.time_embed(t))
        # Fused conditioning vector (time + cache summary).
        c = t_emb + self.cond_proj(cond)

        # Lift latent into hidden width.
        h = self.input_proj(z)  # (B, H, Tz, hidden_dim)

        for i in range(self.num_layers):
            gamma_beta = self.adaln_mods[i](c)  # (B, H, Tz, 2*hidden_dim)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            # AdaLN-Zero: gamma * LayerNorm(h) + beta.
            h = gamma * self.norms[i](h) + beta
            # Main transformation (Linear + SiLU on all but the last layer).
            h = self.linears[i](h)
            if i < self.num_layers - 1:
                h = F.silu(h)

        return self.out_proj(h)  # velocity: (B, H, Tz, d_latent)


class StillCompactorFlowKV(StillCompactor):
    """K/V-split flow-matching compactor with AdaLN-Zero velocity fields.

    Builds on `StillCompactorFlow` with two research-driven changes:

    * **Split latent bank.** The latent is partitioned into a key half
      (the first `head_dim` channels) and a value half (the last `head_dim`
      channels). Each half is transported by its own velocity field, so the
      key flow and the value flow never have to share capacity.

    * **Asymmetric capacity.** The key velocity field is small
      (`key_velocity_hidden`, default `2 * head_dim`) because keys live in a
      low-rank ~3-4% subspace. The value velocity field is larger
      (`val_velocity_hidden`, default `4 * head_dim`) because values are
      higher-rank (d_eff ~ 40-55).

    * **Per-side conditioning.** The key velocity field only sees statistics
      of the key cache; the value velocity field only sees statistics of the
      value cache, so each flow is specialised to its own signal.

    * **AdaLN-Zero.** Both velocity fields are `AdaLNVelocityNet` instances,
      giving the documented DiT/SD3 scaling recipe and an exactly-identity
      flow at initialisation.

    The public `forward` signature is identical to `StillCompactor`, so this
    module is a drop-in replacement inside `StillLM`.
    """

    def __init__(
        self,
        *args,
        key_velocity_hidden: int | None = None,
        val_velocity_hidden: int | None = None,
        flow_steps: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        d = self.head_dim
        if self.d_latent != 2 * d:
            raise ValueError(
                f"StillCompactorFlowKV requires d_latent = 2 * head_dim "
                f"(got d_latent={self.d_latent}, head_dim={d}); the latent "
                "bank must split cleanly into a key half and a value half."
            )

        self.flow_steps = flow_steps
        self.key_velocity_hidden = key_velocity_hidden or 2 * d
        self.val_velocity_hidden = val_velocity_hidden or 4 * d

        # Per-side conditioning dim: mean + max pooling of the relevant cache
        # gives a 2 * head_dim summary vector.
        d_cond = 2 * d

        # Dedicated velocity fields. Smaller for keys, larger for values.
        self.velocity_k = AdaLNVelocityNet(
            d_latent=d,
            d_cond=d_cond,
            hidden_dim=self.key_velocity_hidden,
        )
        self.velocity_v = AdaLNVelocityNet(
            d_latent=d,
            d_cond=d_cond,
            hidden_dim=self.val_velocity_hidden,
        )

        # Context projections: pool the relevant half of the cache into the
        # d_cond conditioning vector used by each flow.
        self.context_proj_k = nn.Linear(2 * d, d_cond)
        self.context_proj_v = nn.Linear(2 * d, d_cond)

        self._init_kv_flow()

    def _init_kv_flow(self) -> None:
        """Zero-init context projections so the initial flow is exactly identity.

        `AdaLNVelocityNet` already zero-inits its modulation and output layers,
        so the velocity is zero at init; zeroing the context projections keeps
        the conditioning signal dormant as well for a clean start.
        """
        with torch.no_grad():
            nn.init.zeros_(self.context_proj_k.weight)
            nn.init.zeros_(self.context_proj_k.bias)
            nn.init.zeros_(self.context_proj_v.weight)
            nn.init.zeros_(self.context_proj_v.bias)

    def _cache_stats(self, cache: torch.Tensor) -> torch.Tensor:
        """Mean + max pooling of a cache tensor along the sequence axis.

        cache: (B, H, T, d)
        returns: (B, H, 2*d)
        """
        mean = cache.mean(dim=2)  # (B, H, d)
        mx = cache.max(dim=2).values  # (B, H, d)
        return torch.cat([mean, mx], dim=-1)  # (B, H, 2*d)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor | None = None,
        return_compact_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compact a per-layer KV cache with K/V-split flow matching.

        keys:    (B, H, T, d) -- RoPE-rotated cached keys.
        values:  (B, H, T, d) -- cached values.
        positions: (T,) -- RoPE positions the keys were rotated at. If None,
                    uses 0..T-1.
        return_compact_cache: if True, also return (Ck, Cv) re-rotated at
                              evenly-spaced output positions.

        Returns a dict with 'latent_out', 'Ck_raw', 'Cv', and (optionally)
        'compact_keys', 'compact_values', 'out_positions' -- the same contract
        as `StillCompactor.forward`.
        """
        B, H, T, d = keys.shape
        if positions is None:
            positions = torch.arange(T, device=keys.device, dtype=torch.float32)

        # Position-free keys, then concatenate [K_free; V] as the cross-attention input.
        keys_free = _inverse_rope(keys, positions, d, base=self.base_rope_base)
        x = torch.cat([keys_free, values], dim=-1)  # (B, H, T, 2d)

        # Expand the learnable latent bank to the batch.
        z = self.latents.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, t, 2d)

        # Step 1: standard Perceiver cross-attention read from block 0. This
        # gives the flow a sensible starting state, identical to StillCompactorFlow.
        block0 = self.blocks[0]
        z = z + block0.cross_attn(block0.norm1(z), x, key_positions=positions)
        z = z + block0.self_attn(block0.norm2(z))
        if block0.use_ffn:
            z = z + block0.ffn(block0.norm3(z))

        # Step 2: K/V-split flow-matching refinement.
        # Per-side conditioning from cache statistics only.
        k_stats = self._cache_stats(keys_free)  # (B, H, 2*d)
        v_stats = self._cache_stats(values)  # (B, H, 2*d)
        cond_k = self.context_proj_k(
            k_stats.unsqueeze(2).expand(-1, -1, self.compact_len, -1)
        )  # (B, H, t, d_cond)
        cond_v = self.context_proj_v(
            v_stats.unsqueeze(2).expand(-1, -1, self.compact_len, -1)
        )  # (B, H, t, d_cond)

        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t_val = torch.tensor(step * dt, device=z.device, dtype=z.dtype)
            t_full = t_val.expand(B, H, self.compact_len)  # (B, H, t)

            # Split latent into key / value halves.
            z_k = z[..., :d].contiguous()  # (B, H, t, d)
            z_v = z[..., d:].contiguous()  # (B, H, t, d)

            # Per-side velocity fields.
            vk = self.velocity_k(z_k, t_full, cond_k)  # (B, H, t, d)
            vv = self.velocity_v(z_v, t_full, cond_v)  # (B, H, t, d)

            # Euler step on each half, then re-concatenate.
            z = torch.cat([z_k + vk * dt, z_v + vv * dt], dim=-1)  # (B, H, t, 2d)

        # Step 3: remaining Perceiver blocks for final refinement.
        for block in self.blocks[1:]:
            z = block(z, x, key_positions=positions)

        # Project to compact keys and values (inherited from StillCompactor).
        Ck = self.key_proj(z)  # (B, H, t, d)
        Cv = self.val_proj(z)  # (B, H, t, d)

        result: dict[str, torch.Tensor] = {"latent_out": z, "Ck_raw": Ck, "Cv": Cv}

        if return_compact_cache:
            # Re-rotate compact keys at evenly spaced output positions.
            out_positions = _even_positions(self.compact_len, T, keys.device)
            Ck_rotated = _apply_rope(Ck, out_positions, d, base=self.base_rope_base)
            result["compact_keys"] = Ck_rotated
            result["compact_values"] = Cv
            result["out_positions"] = out_positions

        return result
