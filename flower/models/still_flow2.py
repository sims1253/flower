"""MeanFlow and OT-CFM flow-matching variants of the Still KV-cache compactor.

Two research variants implemented in a single new module (``still_flow2.py``):

1. ``StillCompactorMeanFlow`` -- replaces the multi-step Euler integration of
   ``StillCompactorFlow`` with a single-step *average-velocity* prediction
   (MeanFlow, Geng et al. 2025, arXiv:2505.13447). At inference the compact
   latent is produced by one forward pass of a ``MeanFlowNet``::

       z_1 = z_0 + u(z_0, t=0) * (1 - 0) = z_0 + u(z_0, t=0)

   A self-consistency loss over a short Euler path (``meanflow_steps`` steps)
   trains the average-velocity field to be straight, which is what makes the
   one-step prediction faithful to the full integration.

2. ``StillCompactorFlowOT`` -- keeps the Euler-integrated velocity field of
   ``StillCompactorFlow`` but replaces the uniform mean/max conditioning with an
   optimal-transport-weighted summary (OT-CFM, Tong et al. 2024,
   arXiv:2302.00482). A Sinkhorn plan between the latent bank and the KV cache
   pairs each latent with its nearest cache cluster, straightening the flow
   trajectories the velocity field has to learn.

Both classes share the ``StillCompactor`` interface::

    forward(keys, values, positions=None, return_compact_cache=False) -> dict

and emit the same keys (``'latent_out'``, ``'Ck_raw'``, ``'Cv'``, and optionally
``'compact_keys'``, ``'compact_values'``, ``'out_positions'``).
``StillCompactorMeanFlow`` additionally emits ``'meanflow_loss'`` (a scalar
consistency loss) during training (a zero scalar at eval).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from flower.models.still import (
    CompactorBlock,
    CompactorCrossAttention,
    CompactorSelfAttention,
    FlowVelocityNet,
    RMSNorm,
    StillCompactor,
    StillCompactorFlow,
    _apply_rope,
    _even_positions,
    _inverse_rope,
    _l2_normalize,
    _sinkhorn_ot,
)


class MeanFlowNet(nn.Module):
    """Average-velocity field for MeanFlow compaction.

    Given the current latent state ``z_t``, the flow time ``t``, a pooled cache
    summary ``cond`` and the full position-free cache ``cache_x``, predicts the
    *average* velocity ``u(t)`` over the interval ``[t, 1]`` such that the
    endpoint is ``z_1 = z_t + u(t) * (1 - t)``.

    Architecture (per the MeanFlow spec for this project):
      * AdaLN-Zero time conditioning -- the time embedding (fused with the
        pooled cache summary) produces a per-layer ``(scale, shift)`` modulation
        applied to every normalised activation. The "Zero" comes from the
        zero-initialised final MLP layer.
      * Cross-attention into the full cache -- lets the velocity field attend to
        the detailed KV structure rather than only the pooled summary.
      * 3-layer SiLU MLP -- ``fc1`` (d_latent -> h), ``fc2`` (h -> h),
        ``fc3`` (h -> d_latent).
      * Zero-initialised final layer -- the field outputs zero velocity at init,
        so the flow starts as the identity.
    """

    def __init__(
        self,
        d_latent: int,
        d_cond: int,
        head_dim: int,
        hidden_dim: int | None = None,
        num_ca_heads: int = 1,
        rope_base: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.d_cond = d_cond
        self.head_dim = head_dim
        h = hidden_dim or 2 * d_latent
        self.h = h

        # Time + cache-summary embedding (the source of every AdaLN modulation).
        self.time_embed = nn.Sequential(
            nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h),
        )
        self.cond_proj = nn.Linear(d_cond, h, bias=False)

        # AdaLN-Zero modulation heads: (scale, shift) for the cross-attention
        # read and for each of the three MLP layer inputs.
        self.adaLN_ca = nn.Linear(h, 2 * d_latent)
        self.adaLN_l1 = nn.Linear(h, 2 * d_latent)
        self.adaLN_l2 = nn.Linear(h, 2 * h)
        self.adaLN_l3 = nn.Linear(h, 2 * h)

        # Cross-attention into the full (position-free) cache.
        self.cross_norm = RMSNorm(d_latent)
        self.cross_attn = CompactorCrossAttention(
            d_latent, head_dim, num_ca_heads, rope_base=rope_base,
        )

        # 3-layer SiLU MLP (pre-norm AdaLN modulation at each layer input).
        self.norm1 = RMSNorm(d_latent)
        self.norm2 = RMSNorm(h)
        self.norm3 = RMSNorm(h)
        self.fc1 = nn.Linear(d_latent, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, d_latent)

        self._init_zero()

    def _init_zero(self) -> None:
        """Zero-init the final layer so the flow starts as identity (u = 0)."""
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        cache_x: torch.Tensor,
        cache_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the average velocity ``u(t)``.

        Args:
            z:               (B, H, t_len, d_latent)  -- current latent state.
            t:               scalar or (B, H, t_len)   -- current flow time in [0, 1).
            cond:            (B, H, t_len, d_cond)     -- pooled mean+max cache summary.
            cache_x:         (B, H, T, 2 * head_dim)   -- position-free [K; V] cache.
            cache_positions: (T,)                       -- RoPE positions for cache keys.

        Returns:
            (B, H, t_len, d_latent) -- average velocity u(t).
        """
        # Broadcast scalar time to the latent grid.
        t = t.expand(z.shape[0], z.shape[1], z.shape[2])

        # Time embedding fused with the pooled cache summary.
        t_emb = self.time_embed(t.unsqueeze(-1))            # (B, H, t_len, h)
        base = t_emb + self.cond_proj(cond)                 # (B, H, t_len, h)

        # Per-layer AdaLN modulations.
        s_ca, sh_ca = self.adaLN_ca(base).chunk(2, dim=-1)
        s1, sh1 = self.adaLN_l1(base).chunk(2, dim=-1)
        s2, sh2 = self.adaLN_l2(base).chunk(2, dim=-1)
        s3, sh3 = self.adaLN_l3(base).chunk(2, dim=-1)

        # Cross-attention read into the full cache (AdaLN-modulated pre-norm).
        z_ca = self.cross_norm(z) * (1.0 + s_ca) + sh_ca
        z = z + self.cross_attn(z_ca, cache_x, key_positions=cache_positions)

        # 3-layer SiLU MLP with AdaLN-Zero per-layer modulation.
        h1 = F.silu(self.fc1(self.norm1(z) * (1.0 + s1) + sh1))
        h2 = F.silu(self.fc2(self.norm2(h1) * (1.0 + s2) + sh2))
        out = self.fc3(self.norm3(h2) * (1.0 + s3) + sh3)
        return out


class StillCompactorMeanFlow(StillCompactor):
    """MeanFlow compactor: one-step average-velocity KV compaction.

    Replaces the ``flow_steps``-deep Euler integration of ``StillCompactorFlow``
    with a single MeanFlow forward pass at inference. During training a short
    (``meanflow_steps``) Euler path is still rolled out -- purely to materialise
    intermediate states ``z_t`` and the integrated endpoint ``z_target`` for the
    MeanFlow consistency loss::

        L = || u_pred(t) - (z_target - z_t) / (1 - t) ||^2

    which trains the average-velocity field to be straight. Once straight, the
    one-step endpoint ``z_0 + u(z_0, t=0)`` reproduces the full integration, so
    inference collapses to a single network call.

    The block structure mirrors ``StillCompactorFlow``: cross-attention read
    from the first Perceiver block, the (mean+max) pooled conditioning, then the
    remaining Perceiver blocks applied after the flow.
    """

    def __init__(self, *args, meanflow_steps: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.meanflow_steps = max(1, int(meanflow_steps))

        d = self.head_dim
        d_cond = 4 * d  # mean + max of the 2*d cache, consistent with StillCompactorFlow.

        self.meanflow_net = MeanFlowNet(
            d_latent=self.d_latent,
            d_cond=d_cond,
            head_dim=d,
            hidden_dim=2 * self.d_latent,
            num_ca_heads=1,
            rope_base=self.rope_base,
        )

    def _pool_cond(self, x: torch.Tensor) -> torch.Tensor:
        """Mean + max pool the cache into a per-latent conditioning vector.

        Args:
            x: (B, H, T, 2d) -- position-free [K; V].

        Returns:
            (B, H, t_len, 4d) -- broadcast pooled summary, one per latent slot.
        """
        t_len = self.compact_len
        mean_pool = x.mean(dim=2)                        # (B, H, 2d)
        max_pool = x.amax(dim=2)                         # (B, H, 2d)
        cond = torch.cat([mean_pool, max_pool], dim=-1)  # (B, H, 4d)
        return cond.unsqueeze(2).expand(-1, -1, t_len, -1)

    def _consistency_loss(
        self,
        z0: torch.Tensor,
        cond: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Roll out a short Euler path and regress the average velocity.

        The path is computed under ``no_grad`` (it only supplies detached
        targets ``z_t`` / ``z_target``). A single random intermediate time ``t``
        is sampled per call; the field is regressed so that
        ``u(z_t, t) == (z_target - z_t) / (1 - t)``, the MeanFlow identity for a
        straight path.

        Args:
            z0:        (B, H, t_len, d_latent) -- latent state entering the flow.
            cond:      (B, H, t_len, d_cond)   -- pooled mean+max conditioning.
            x:         (B, H, T, 2d)           -- position-free [K; V] cache.
            positions: (T,)                    -- cache RoPE positions.

        Returns:
            Scalar MSE consistency loss.
        """
        B, H, t_len, _ = z0.shape
        dt = 1.0 / self.meanflow_steps

        # Roll out the integration path (no graph -- only targets are needed).
        # The average velocity is used as the step velocity: when u is the true
        # average velocity this yields a straight line z_t = z_0 + u * t, which
        # makes the consistency target below exact and self-consistent.
        with torch.no_grad():
            z_cur = z0.detach()
            z_path = [z_cur]  # z_path[k] is the state at time k * dt.
            for step in range(self.meanflow_steps):
                t_val = step * dt
                t_t = torch.full((B, H, t_len), t_val, device=z0.device, dtype=z0.dtype)
                u_step = self.meanflow_net(z_cur, t_t, cond, x, cache_positions=positions)
                z_cur = z_cur + u_step * dt
                z_path.append(z_cur)
            z_target = z_cur  # integrated endpoint (state at t = 1).

        # Sample one intermediate time and enforce the MeanFlow identity there.
        k = int(torch.randint(0, self.meanflow_steps, (), device=z0.device).item())
        t_k = k * dt
        denom = max(1.0 - t_k, 1e-6)
        z_tk = z_path[k]                                    # detached state at t_k
        t_k_t = torch.full((B, H, t_len), t_k, device=z0.device, dtype=z0.dtype)
        u_pred = self.meanflow_net(z_tk, t_k_t, cond, x, cache_positions=positions)
        target = (z_target - z_tk) / denom                  # detached average velocity
        return F.mse_loss(u_pred, target)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor | None = None,
        return_compact_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compact a per-layer KV cache via single-step MeanFlow.

        Args:
            keys:                (B, H, T, d) -- RoPE-rotated cached keys.
            values:              (B, H, T, d) -- cached values.
            positions:           (T,)         -- RoPE positions (default 0..T-1).
            return_compact_cache: if True, also return re-rotated compact keys/values.
        """
        B, H, T, d = keys.shape
        if positions is None:
            positions = torch.arange(T, device=keys.device, dtype=torch.float32)

        keys_free = _inverse_rope(keys, positions, d, base=self.base_rope_base)
        x = torch.cat([keys_free, values], dim=-1)          # (B, H, T, 2d)
        t_len = self.compact_len

        # Initial latents.
        z0 = self.latents.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, t_len, d_latent)

        # Step 1: cross-attention read from the first Perceiver block.
        block0 = self.blocks[0]
        z0 = z0 + block0.cross_attn(block0.norm1(z0), x, key_positions=positions)
        z0 = z0 + block0.self_attn(block0.norm2(z0))
        if block0.use_ffn:
            z0 = z0 + block0.ffn(block0.norm3(z0))

        # Pooled conditioning (mean + max of the cache), reused everywhere.
        cond = self._pool_cond(x)

        # Step 2: single-step MeanFlow prediction -- the inference form, used in
        # both train and eval so the KL distillation directly trains the
        # one-step endpoint that is used at inference.
        #     z_out = z0 + u(z0, t=0) * (1 - 0) = z0 + u(z0, t=0).
        t_zero = torch.zeros((B, H, t_len), device=z0.device, dtype=z0.dtype)
        u0 = self.meanflow_net(z0, t_zero, cond, x, cache_positions=positions)
        z_out = z0 + u0

        # Step 3: MeanFlow consistency loss (training only).
        if self.training:
            meanflow_loss = self._consistency_loss(z0, cond, x, positions)
        else:
            meanflow_loss = z0.new_zeros(())

        # Step 4: remaining Perceiver blocks for final refinement.
        for block in self.blocks[1:]:
            z_out = block(z_out, x, key_positions=positions)

        # Step 5: project to compact keys / values.
        Ck = self.key_proj(z_out)
        Cv = self.val_proj(z_out)

        result: dict[str, torch.Tensor] = {
            "latent_out": z_out,
            "Ck_raw": Ck,
            "Cv": Cv,
            "meanflow_loss": meanflow_loss,
        }
        if return_compact_cache:
            out_positions = _even_positions(self.compact_len, T, keys.device)
            Ck_rotated = _apply_rope(Ck, out_positions, d, base=self.base_rope_base)
            result["compact_keys"] = Ck_rotated
            result["compact_values"] = Cv
            result["out_positions"] = out_positions
        return result


class StillCompactorFlowOT(StillCompactorFlow):
    """OT-CFM-coupled flow compactor.

    Identical to ``StillCompactorFlow`` (Euler-integrated velocity field) except
    the per-latent conditioning is weighted by a Sinkhorn optimal-transport plan
    between the latent bank and the KV cache, instead of uniform mean/max
    pooling. Pairing each latent with its nearest cache cluster straightens the
    flow trajectories the velocity field must learn (OT-CFM, Tong et al. 2024).

    The transport plan is computed from a cosine-distance cost between the
    initial latent bank and the position-free cache entries, then used to
    produce an OT-weighted cache summary that feeds the existing
    ``context_proj_k`` / ``context_proj_v`` conditioning heads inherited from
    ``StillCompactorFlow``.
    """

    def __init__(
        self,
        *args,
        ot_epsilon: float = 0.1,
        ot_iters: int = 10,
        **kwargs,
    ) -> None:
        # ``ot_epsilon`` / ``ot_iters`` flow through to ``StillCompactor.__init__``,
        # which stores them as ``self.ot_epsilon`` / ``self.ot_iters``.
        super().__init__(*args, ot_epsilon=ot_epsilon, ot_iters=ot_iters, **kwargs)
        # The OT cost is computed directly between the latent bank and the cache
        # (see ``_ot_conditioning``), which requires them to share a dimension.
        # This is the canonical Still config (d_latent == 2 * head_dim), already
        # assumed by the identity initialisation in ``StillCompactor``.
        if self.d_latent != 2 * self.head_dim:
            raise ValueError(
                "StillCompactorFlowOT requires d_latent == 2 * head_dim "
                f"(got d_latent={self.d_latent}, head_dim={self.head_dim}) so "
                "the latent bank and the [K; V] cache share a dimension for the "
                "OT cost matrix."
            )

    def _ot_conditioning(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """OT-weighted per-latent conditioning.

        Args:
            x: (B, H, T, 2d) -- position-free [K; V].

        Returns:
            (cond_k, cond_v), each (B, H, t_len, 4d), the OT-weighted summaries
            fed to the velocity-field FiLM heads.
        """
        B, H, T, _ = x.shape
        t_len = self.compact_len

        # Cost matrix + Sinkhorn plan are a fixed geometric coupling given the
        # current latents, so they are computed under no_grad. Gradients still
        # reach the cache (and thus the velocity field) through the OT-weighted
        # read ``cond_ot = plan @ x`` below.
        with torch.no_grad():
            lat = self.latents.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, t_len, d_latent)
            lat_n = _l2_normalize(lat)
            x_n = _l2_normalize(x)
            cost = 1.0 - torch.einsum(
                "bhid,bhjd->bhij", lat_n.float(), x_n.float()
            )  # (B, H, t_len, T)
            plan = _sinkhorn_ot(
                cost.reshape(B * H, t_len, T),
                epsilon=self.ot_epsilon,
                n_iters=self.ot_iters,
            ).reshape(B, H, t_len, T).to(x.dtype)

        # OT-weighted cache summary: each latent reads its transported mass.
        cond_ot = torch.einsum("bhiT,bhTd->bhid", plan, x)    # (B, H, t_len, 2d)

        cond_k = self.context_proj_k(cond_ot)                 # (B, H, t_len, 4d)
        cond_v = self.context_proj_v(cond_ot)
        return cond_k, cond_v

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor | None = None,
        return_compact_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compact a per-layer KV cache via OT-coupled flow matching.

        Args:
            keys:                (B, H, T, d) -- RoPE-rotated cached keys.
            values:              (B, H, T, d) -- cached values.
            positions:           (T,)         -- RoPE positions (default 0..T-1).
            return_compact_cache: if True, also return re-rotated compact keys/values.
        """
        B, H, T, d = keys.shape
        if positions is None:
            positions = torch.arange(T, device=keys.device, dtype=torch.float32)

        keys_free = _inverse_rope(keys, positions, d, base=self.base_rope_base)
        x = torch.cat([keys_free, values], dim=-1)            # (B, H, T, 2d)
        t_len = self.compact_len

        # Step 1: cross-attention read from the first Perceiver block.
        z = self.latents.unsqueeze(0).expand(B, -1, -1, -1)
        block0 = self.blocks[0]
        z = z + block0.cross_attn(block0.norm1(z), x, key_positions=positions)
        z = z + block0.self_attn(block0.norm2(z))
        if block0.use_ffn:
            z = z + block0.ffn(block0.norm3(z))

        # Step 2: OT-weighted conditioning (replaces uniform mean/max pooling).
        cond_k, cond_v = self._ot_conditioning(x)

        # Step 3: Euler integration of the velocity field (same as StillCompactorFlow).
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t_val = torch.tensor(step * dt, device=z.device, dtype=z.dtype)
            d_k = self.velocity_keys(z, t_val.expand(B, H, t_len), cond_k)
            d_v = self.velocity_vals(z, t_val.expand(B, H, t_len), cond_v)
            velocity = 0.5 * (d_k + d_v)
            z = z + velocity * dt

        # Step 4: remaining Perceiver blocks.
        for block in self.blocks[1:]:
            z = block(z, x, key_positions=positions)

        # Step 5: project to compact keys / values.
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


__all__ = [
    "MeanFlowNet",
    "StillCompactorMeanFlow",
    "StillCompactorFlowOT",
    # Re-exported helpers / base classes for convenience.
    "StillCompactor",
    "StillCompactorFlow",
    "FlowVelocityNet",
    "CompactorCrossAttention",
    "CompactorSelfAttention",
    "CompactorBlock",
    "RMSNorm",
    "_apply_rope",
    "_inverse_rope",
    "_even_positions",
    "_l2_normalize",
    "_sinkhorn_ot",
]
