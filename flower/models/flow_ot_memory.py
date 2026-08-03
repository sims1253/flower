"""Flow-OT memory: Sinkhorn-coupled, flow-matched memory writes.

Idea (novel to Flower's bake-off): instead of broadcasting the same update to all
memory slots (summary_memory) or hashing tokens to slots with fixed-functional
addressing (Bloom / Neural Bloom Filter), we treat the write step as an *optimal
transport* problem from a small set of perceiver-style "source" summaries of the
current window to the existing memory slots, then apply a *flow-matched* velocity
field that uses the layer depth as the integration time.

Pipeline at each block:

  1. Perceiver compression: cross-attention from `ot_source_points` learnable
     queries into the token window -> source set S of shape (B, P, D).
  2. Cost matrix: 1 - cosine_similarity between projected source and projected
     slot embeddings, shape (B, P, N_slots).
  3. Sinkhorn (entropic OT) with rectangular uniform marginals (1/P on rows,
     1/N on columns) -> transport plan P of the same shape.
  4. Write signal per slot: column-normalised plan applied to the source set ->
     each slot receives a content-weighted mix of source points (no false
     negatives: every source point routes mass somewhere; some false positives
     are accepted, the velocity field learns to filter them).
  5. Flow-matched update: a velocity net v_theta(memory, write_signal, t) with
     scalar time t = (layer_idx + 1) / num_layers. Each layer is one Euler step
     of the same conceptual flow; depth literally becomes integration time.

This composes the project's two themes -- flow matching and summary networks --
with a structured addressing mechanism that adapts per-forward-pass instead of
the static hashing used in classic Bloom filters. None of the existing variants
(flow_meanflow, flow_pma, titans_mac, summary_memory, ...) combine OT routing
with a layer-as-time velocity field.

Cost: the Sinkhorn loop is O(B * P * N_slots * iters); with P=16, N_slots=16,
iters=10 that is ~25k flops per layer-batch -- negligible vs. local attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward
from flower.models.memory import MemoryRead


def _sinkhorn_rect(cost: torch.Tensor, *, epsilon: float, iters: int) -> torch.Tensor:
    """Entropic OT plan for rectangular cost (B, N, M) with uniform marginals.

    Returns a non-negative tensor of the same shape whose rows sum to 1/N and
    whose columns sum to 1/M (approximately, after `iters` Sinkhorn updates).
    Computation runs in fp32 to avoid logsumexp underflow under bf16.
    """
    bsz, n_rows, n_cols = cost.shape
    log_p = (-cost / max(epsilon, 1e-6)).float()
    log_a = math.log(1.0 / n_rows)
    log_b = math.log(1.0 / n_cols)
    log_u = torch.zeros(bsz, n_rows, device=cost.device)
    log_v = torch.zeros(bsz, n_cols, device=cost.device)
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(log_p + log_v.unsqueeze(1), dim=2)
        log_v = log_b - torch.logsumexp(log_p + log_u.unsqueeze(2), dim=1)
    plan = torch.exp(log_p + log_u.unsqueeze(2) + log_v.unsqueeze(1))
    return plan.to(cost.dtype)


class FlowOTMemoryBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = MemoryRead(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout, config=config)

        # Perceiver compression to ot_source_points.
        self.source_queries = nn.Parameter(torch.randn(1, config.ot_source_points, config.d_model) * 0.02)
        self.source_attn = nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)

        # Cost-space projections: separates the "what should this source attach to"
        # subspace from the raw token representation. Without separate projections
        # the cost matrix is dominated by token-position similarity.
        self.source_proj = nn.Linear(config.d_model, config.d_model)
        self.slot_proj = nn.Linear(config.d_model, config.d_model)

        # Flow velocity field: depth-as-time conditioning, plus the write_signal
        # routed in from the OT plan and the current memory state.
        self.time_embed = nn.Linear(1, config.d_model)
        self.velocity = nn.Sequential(
            nn.Linear(3 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], self.config.memory_slots, self.config.d_model)

    def _update_memory(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        queries = self.source_queries.expand(bsz, -1, -1)
        source, _ = self.source_attn(queries, x, x, need_weights=False)  # (B, P, D)

        src_e = F.normalize(self.source_proj(source), dim=-1)
        slot_e = F.normalize(self.slot_proj(memory), dim=-1)
        # Cosine cost in [0, 2]; lower = more similar = preferred transport.
        cost = 1.0 - src_e @ slot_e.transpose(-2, -1)

        plan = _sinkhorn_rect(cost, epsilon=self.config.ot_epsilon, iters=self.config.ot_iters)
        # Re-normalise columns so each slot's incoming distribution sums to 1; the
        # write signal becomes a proper convex combination of source points and
        # the velocity field sees comparable magnitudes regardless of N_slots/P.
        plan_col = plan / (plan.sum(dim=1, keepdim=True) + 1e-9)  # (B, P, S)
        write_signal = plan_col.transpose(1, 2) @ source  # (B, S, D)
        # Diagnostics: column-plan entropy (low = each slot deterministically
        # routed to one source point, high = each slot a uniform mix) and slot
        # mass max (how concentrated source points routed onto a single slot
        # before column renorm -- captures slot-collapse).
        with torch.no_grad():
            entropy = -(plan_col.clamp_min(1e-9) * plan_col.clamp_min(1e-9).log()).sum(dim=1).mean()
            slot_mass = plan.sum(dim=1)  # (B, S) mass arriving at each slot
            self.last_diag_ot_plan_entropy = float(entropy.cpu())
            self.last_diag_ot_slot_mass_max = float(slot_mass.max().cpu())

        t = float(self.layer_idx + 1) / float(max(1, self.config.num_layers))
        t_emb = self.time_embed(memory.new_full((bsz, 1, 1), t)).expand(-1, memory.shape[1], -1)
        delta = self.velocity(torch.cat([memory, write_signal, t_emb], dim=-1))
        return memory + delta / float(max(1, self.config.num_layers))

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory is None:
            memory = self._initial_memory(x)
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))
        memory = self._update_memory(memory, x)
        return x, memory


def build_flow_ot_memory_model(config: ModelConfig) -> CausalLM:
    blocks = [FlowOTMemoryBlock(config, i) for i in range(config.num_layers)]
    return CausalLM(config, blocks)
