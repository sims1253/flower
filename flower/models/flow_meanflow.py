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
from flower.models.base import CausalSelfAttention, FeedForward
from flower.models.memory import MemoryRead


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
        time = self.time(torch.stack([t, r], dim=-1))  # (B, D)
        cond_emb = self.cond_proj(cond)  # (B, D)
        bias = (time + cond_emb).unsqueeze(1)  # (B, 1, D)
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
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout)

        flow_hidden = max(128, config.d_model)
        self.field = MeanFlowField(
            slot_dim=config.d_model,
            cond_dim=config.d_model,
            hidden_dim=flow_hidden,
        )

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], self.config.memory_slots, self.config.d_model)

    def _meanflow_loss(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        # Sample r <= t in [0, 1] per batch element. (Independent samples reduce
        # variance vs. a single shared (t, r) for the whole batch.)
        bsz = z0.shape[0]
        device = z0.device
        u = torch.rand(bsz, device=device)
        v = torch.rand(bsz, device=device)
        t = torch.maximum(u, v)
        r = torch.minimum(u, v)

        # Optional OT-CFM coupling: pair z0 with z1 across the batch.
        z1_paired = z1
        if self.config.meanflow_ot_cfm and bsz > 1:
            with torch.no_grad():
                # Cost = squared distance between flattened slot tensors. Cheap and
                # invariant to slot order would require Sinkhorn over slots first;
                # we accept the per-batch pairing as the standard OT-CFM choice.
                flat_z0 = z0.reshape(bsz, -1)
                flat_z1 = z1.reshape(bsz, -1)
                cost = torch.cdist(flat_z0, flat_z1, p=2.0).pow(2)
                plan = _sinkhorn_plan(
                    cost,
                    epsilon=self.config.meanflow_ot_epsilon,
                    iters=self.config.meanflow_ot_iters,
                )
            # Soft re-mixing of z1 according to the transport plan. The B factor
            # restores per-row marginal mass (Sinkhorn enforces 1/B marginals).
            z1_paired = bsz * torch.einsum("ij,jsd->isd", plan, z1)

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
        if memory is None:
            memory = self._initial_memory(x)
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))

        cond = x.mean(dim=1)
        z0 = memory
        # 1-step inference always (train + eval): M_new = M_old + u_theta(M_old, 1, 0, cond).
        # The LM cross-entropy backprops into the field via this endpoint use, so the
        # field's "what should memory become?" signal is the LM loss itself.
        ones = x.new_ones(())
        zeros = x.new_zeros(())
        u_endpoint = self.field(z0, ones, zeros, cond)
        new_memory = z0 + u_endpoint

        aux = None
        if self.training:
            # MeanFlow consistency: intermediate (t, r) predictions should match the
            # endpoint velocity along the straight path. Target is the endpoint
            # prediction with stop-grad, so the aux loss is a self-distillation of
            # the field across the (t, r) interval — no separate "target network".
            z1 = (z0 + u_endpoint).detach()
            aux = self._meanflow_loss(z0, z1, cond)
        return x, new_memory, aux


class MeanFlowMemoryLM(nn.Module):
    """CausalLM variant that adds MeanFlow auxiliary losses to the main loss."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        from dataclasses import asdict

        self.config = config
        self._asdict = asdict
        self.token = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([MeanFlowMemoryBlock(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, object]:
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input length exceeds max_seq_len")
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token(input_ids) + self.pos(pos).unsqueeze(0)
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
        diagnostics = {
            "parameter_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "config": self._asdict(self.config),
            "meanflow_aux_loss": float(torch.stack(aux_losses).mean().detach().cpu()) if aux_losses else 0.0,
        }
        return {"logits": logits, "loss": loss, "diagnostics": diagnostics}


def build_flow_meanflow_model(config: ModelConfig) -> nn.Module:
    return MeanFlowMemoryLM(config)
