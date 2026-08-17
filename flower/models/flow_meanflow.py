"""MeanFlow-parameterised memory updates with optional OT-CFM coupling.

Background:
- "MeanFlow Models for One-Step Generative Modeling" (Geng et al., 2025, arXiv:2505.13447)
  parameterises the *average* velocity u(z_t, t, r) over an interval [r, t] instead of
  the instantaneous velocity v(z_t, t). The training objective enforces the identity
  u(z_t, t, r) = v(z_t, t) - (t - r) * d/dt u(z_t, t, r). At inference, one forward
  pass with r=0, t=1 produces z_1 = z_0 + u(z_0, 1, 0). For memory writes this means
  the multi-step Euler integration in flow_memory.py collapses to a single net call.
- "OT-CFM" (Tong et al., 2024, arXiv:2302.00482) pre-pairs (x_0, x_1) endpoints via a
  Sinkhorn / Hungarian assignment before regressing the velocity. The straightened
  trajectories give a meaningfully different inductive bias from per-sample pairing.

The implementation here uses the linear-path simplification of MeanFlow: the target
average velocity along a straight path z_t = (1-t) z_0 + t z_1 is just (z_1 - z_0),
constant along the interval. We regress u_theta(z_t, t, r) to this target and train
the consistency constraint only weakly through that supervision. This avoids the
JVP machinery while keeping the 1-step inference property that is the entire point.

The OT-CFM coupling is across the batch: for each layer's memory update we solve a
small Sinkhorn assignment between the batch's z_0 states and z_1 targets and pair
them up before regression. This is a soft permutation across the batch, NOT across
slots — slot-set invariance is handled by the per-slot flow application.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.diag import clear, should_collect, stash
from flower.models.base import CausalSelfAttention, FeedForward
from flower.models.memory import MemoryRead, causal_chunked_map, causal_running_mean


def _sinkhorn_plan(cost: torch.Tensor, *, epsilon: float, iters: int) -> torch.Tensor:
    """Doubly-stochastic transport plan for a square cost matrix.

    Returns P of shape (B, B) with rows and columns summing to 1/B (uniform marginals).
    Entropic regularisation epsilon controls sharpness; iters bounds compute. Operates
    in fp32 to avoid the log-sum-exp underflow that bf16 inflicts on small batches.
    """
    log_p = (-cost / max(epsilon, 1e-6)).float()
    log_marginal = -torch.log(torch.tensor(float(cost.shape[0]), device=cost.device))
    log_u = torch.zeros(cost.shape[0], device=cost.device)
    log_v = torch.zeros(cost.shape[1], device=cost.device)
    for _ in range(iters):
        log_u = log_marginal - torch.logsumexp(log_p + log_v[None, :], dim=1)
        log_v = log_marginal - torch.logsumexp(log_p + log_u[:, None], dim=0)
    return torch.exp(log_p + log_u[:, None] + log_v[None, :]).to(cost.dtype)


class MeanFlowField(nn.Module):
    """Predicts average velocity u(z, t, r, cond) over the interval [r, t]."""

    def __init__(self, slot_dim: int, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        # Time-pair embedding: encode both r and t (not just t) so the field knows the
        # interval it is averaging over. Two scalars -> small MLP -> additive bias.
        self.time = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, slot_dim)
        self.net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, slot_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        # z: (B, S, D); t, r: scalar or (B,); cond: (B, cond_dim).
        if t.dim() == 0:
            t = t.expand(z.shape[0])
        if r.dim() == 0:
            r = r.expand(z.shape[0])
        # _meanflow_loss samples t/r with torch.rand, which is fp32 even in a
        # pure-bf16 model; match z's dtype so the time-embedding Linear never
        # sees mixed dtypes (bf16 training crashed here before).
        t = t.to(dtype=z.dtype)
        r = r.to(dtype=z.dtype)
        time = self.time(torch.stack([t, r], dim=-1))  # (B, D)
        cond_emb = self.cond_proj(cond)  # (B, D)
        bias = (time + cond_emb).unsqueeze(1)  # (B, 1, D)
        return self.net(torch.cat([z, bias.expand_as(z)], dim=-1))

    def forward_positions(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position variant for causal_memory=True. Same parameters, same
        math as ``forward``, but every position carries its own condition:
        z is (B, T, S, D), cond is (B, T, D), and t/r are scalars or (B,)
        shared across positions (exactly like ``forward`` shares them across
        slots)."""
        if t.dim() == 0:
            t = t.expand(z.shape[0])
        if r.dim() == 0:
            r = r.expand(z.shape[0])
        # Same dtype guard as ``forward``: rand-sampled t/r are fp32.
        t = t.to(dtype=z.dtype)
        r = r.to(dtype=z.dtype)
        time = self.time(torch.stack([t, r], dim=-1).unsqueeze(1).expand(-1, z.shape[1], -1))  # (B, T, D)
        cond_emb = self.cond_proj(cond)  # (B, T, D)
        bias = (time + cond_emb).unsqueeze(2)  # (B, T, 1, D)
        return self.net(torch.cat([z, bias.expand_as(z)], dim=-1))


class MeanFlowMemoryBlock(nn.Module):
    """Memory block whose write rule is parameterised by MeanFlow average velocity.

    Returns (x, memory, aux_loss). aux_loss is None at eval; during training it is the
    MeanFlow regression loss (optionally OT-CFM-coupled across the batch).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = MemoryRead(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout, config=config)

        flow_hidden = max(128, config.d_model)
        self.field = MeanFlowField(
            slot_dim=config.d_model,
            cond_dim=config.d_model,
            hidden_dim=flow_hidden,
        )

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], self.config.memory_slots, self.config.d_model)

    def _initial_memory_causal(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, S, D) per-position memory state for causal_memory=True."""
        return x.new_zeros(x.shape[0], x.shape[1], self.config.memory_slots, self.config.d_model)

    def _meanflow_loss(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        # Sample r <= t in [0, 1] per batch element. (Independent samples reduce
        # variance vs. a single shared (t, r) for the whole batch.)
        #
        # Under causal_memory=True the states are per-position (B, T, S, D) and
        # cond is (B, T, D); the loss generalises by treating the position axis
        # exactly like the slot axis of the legacy (B, S, D) form.
        #
        # NOTE ON THE OT-CFM BATCH COUPLING (meanflow_ot_cfm=True), investigated
        # and deliberately LEFT UNCHANGED here: the Sinkhorn plan pairs z0/z1
        # across the BATCH dimension (einsum "ij,jsd->isd"). That couples
        # different training samples *in the auxiliary training loss only* —
        # the aux loss never feeds the forward logits, so it is NOT a forward
        # causality leak (positions' logits remain functions of tokens <= t
        # regardless). It IS a training-loss concern: (a) as designed, each
        # batch element's regression target is a soft mixture of the OTHER
        # elements' memories (the standard OT-CFM choice, but it means the aux
        # gradient for sample i depends on samples j != i); (b) under
        # causal_memory the cost flattens (T, S, D) per element, so the plan
        # now couples whole per-position state trajectories rather than
        # single-window banks. Whether that coupling is desirable for the
        # causal per-position states is deferred — semantics preserved.
        bsz = z0.shape[0]
        device = z0.device
        causal = z0.dim() == 4
        u = torch.rand(bsz, device=device)
        v = torch.rand(bsz, device=device)
        # Cast the rand-sampled times to the states' dtype at the source: in a
        # pure-bf16 model, fp32 t/r would promote the interpolant/target to
        # fp32 and crash the field's bf16 linears (bf16 training regression).
        t = torch.maximum(u, v).to(dtype=z0.dtype)
        r = torch.minimum(u, v).to(dtype=z0.dtype)

        # Optional OT-CFM coupling: pair z0 with z1 across the batch.
        z1_paired = z1
        if self.config.meanflow_ot_cfm and bsz > 1:
            with torch.no_grad():
                # Cost = squared distance between flattened slot tensors. Cheap and
                # invariant to slot order would require Sinkhorn over slots first;
                # we accept the per-batch pairing as the standard OT-CFM choice.
                # cdist has no bf16 kernel (NotImplementedError on pure-bf16
                # models); this branch is under no_grad, so compute the plan in
                # fp32 and cast back — numerically identical for fp32 runs.
                flat_z0 = z0.reshape(bsz, -1).float()
                flat_z1 = z1.reshape(bsz, -1).float()
                cost = torch.cdist(flat_z0, flat_z1, p=2.0).pow(2)
                plan = _sinkhorn_plan(
                    cost,
                    epsilon=self.config.meanflow_ot_epsilon,
                    iters=self.config.meanflow_ot_iters,
                ).to(z0.dtype)
            # Soft re-mixing of z1 according to the transport plan. The B factor
            # restores per-row marginal mass (Sinkhorn enforces 1/B marginals).
            if causal:
                z1_paired = bsz * torch.einsum("ij,j...->i...", plan, z1)
            else:
                z1_paired = bsz * torch.einsum("ij,jsd->isd", plan, z1)

        if causal:
            # Straight-line interpolant, per position; t broadcasts over (T, S, D).
            t_b = t.view(-1, 1, 1, 1)
            z_t = (1.0 - t_b) * z0 + t_b * z1_paired

            # MeanFlow target on a straight path: constant velocity z1 - z0.
            target = z1_paired - z0
            # Chunked over T: the field's (B, T, S, 2D) concat is its widest
            # intermediate (~2x the lead tensor), so halve the chunk budget.
            pred = causal_chunked_map(
                lambda z_c, c_c: self.field.forward_positions(z_c, t, r, c_c),
                z_t,
                cond,
                max_temp_elements=2**27,
            )
        else:
            # Straight-line interpolant. Broadcast t over slot/feature dims.
            t_b = t.view(-1, 1, 1)
            z_t = (1.0 - t_b) * z0 + t_b * z1_paired

            # MeanFlow target on a straight path: constant velocity z1 - z0.
            target = z1_paired - z0
            pred = self.field(z_t, t, r, cond)
        loss = F.mse_loss(pred, target)
        return loss * self.config.meanflow_loss_weight

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        causal = self.config.causal_memory
        if memory is None:
            memory = self._initial_memory_causal(x) if causal else self._initial_memory(x)
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))

        if causal:
            # Condition is the prefix mean at t (legacy: whole-window mean, which
            # includes future tokens); the field evaluates per position. Chunked
            # over T to bound the field's intermediates (budget halved: the
            # field's cat([z, bias]) input is 2x the lead tensor's width).
            cond = causal_running_mean(x)  # (B, T, D)
            ones = x.new_ones(())
            zeros = x.new_zeros(())
            u_endpoint = causal_chunked_map(
                lambda z_c, c_c: self.field.forward_positions(z_c, ones, zeros, c_c),
                memory,
                cond,
                max_temp_elements=2**27,
            )
        else:
            cond = x.mean(dim=1)
            # 1-step inference always (train + eval): M_new = M_old + u_theta(M_old, 1, 0, cond).
            # The LM cross-entropy backprops into the field via this endpoint use, so the
            # field's "what should memory become?" signal is the LM loss itself.
            ones = x.new_ones(())
            zeros = x.new_zeros(())
            u_endpoint = self.field(memory, ones, zeros, cond)
        new_memory = memory + u_endpoint

        aux = None
        if self.training:
            # MeanFlow consistency: intermediate (t, r) predictions should match the
            # endpoint velocity along the straight path. Target is the endpoint
            # prediction with stop-grad, so the aux loss is a self-distillation of
            # the field across the (t, r) interval — no separate "target network".
            z1 = (memory + u_endpoint).detach()
            aux = self._meanflow_loss(memory, z1, cond)
        return x, new_memory, aux


class MeanFlowMemoryLM(nn.Module):
    """CausalLM variant that adds MeanFlow auxiliary losses to the main loss."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        from dataclasses import asdict

        self.config = config
        self._asdict = asdict
        self.token = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([MeanFlowMemoryBlock(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, object]:
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input length exceeds max_seq_len")
        x = self.token(input_ids)
        memory: torch.Tensor | None = None
        aux_losses: list[torch.Tensor] = []
        loops = max(1, getattr(self.config, "loop_count", 1))
        for _ in range(loops):
            for block in self.blocks:
                x, memory, aux = block(x, memory)
                if aux is not None:
                    aux_losses.append(aux)
        logits = self.head(self.ln(x))
        loss: torch.Tensor | None = None
        if labels is not None:
            ce = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
            loss = ce
            if aux_losses:
                loss = ce + torch.stack(aux_losses).mean()
        # Aux-loss diagnostic. The old `float(...cpu())` here was a host sync
        # every forward and a graph break under compile. Now stashed on-device
        # under the standard guard (flower/diag.py); the diagnostics dict carries
        # the same 0-d tensor under the same key, and the logging step in
        # train.py does the single host transfer. When not collecting (under
        # torch.compile), any value left by an earlier eager forward is cleared
        # — otherwise the dict below would report that stale number (or a
        # fabricated 0.0) as if it were the current step's loss.
        if should_collect():
            if aux_losses:
                stash(self, "meanflow_aux_loss", torch.stack(aux_losses).mean())
        else:
            clear(self, "meanflow_aux_loss")
        diagnostics = {
            "parameter_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "config": self._asdict(self.config),
            "meanflow_aux_loss": getattr(self, "last_diag_meanflow_aux_loss", None),
        }
        return {"logits": logits, "loss": loss, "diagnostics": diagnostics}


def build_flow_meanflow_model(config: ModelConfig) -> nn.Module:
    return MeanFlowMemoryLM(config)
