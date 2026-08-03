"""PMA (Pooling by Multihead Attention) with learned flow on inducing points.

Background:
- Set Transformer (Lee et al., 2019) introduced PMA: a small set of learned latent
  "seed" vectors that pool a token set via multihead attention. The aggregation is
  permutation-invariant and the number of seeds is decoupled from set size.
- Standard PMA uses the attention output directly. The plan here treats the seed
  vectors as inducing points on a manifold and applies a learned conditional flow
  that transports each seed toward an attractor conditioned on the token-pool output.
  The flow is a small ConditionalCouplingFlow (invertible, exact log-det if needed)
  applied per-slot.

This is the direct upgrade of summary_memory's `aggregation = max`: the slot update
is no longer a fixed max-pool followed by an MLP — it is a learned transport from
the existing slot toward the new token information.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.flows.coupling import ConditionalCouplingFlow
from flower.models.base import CausalSelfAttention, FeedForward
from flower.models.memory import MemoryRead


class _PerSlotFlow(nn.Module):
    """Conditional coupling flow applied independently to each slot.

    The slot dimension D may be odd; if so we pad to even, run the flow, and unpad.
    """

    def __init__(self, slot_dim: int, cond_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.padded_dim = slot_dim + (slot_dim % 2)
        self.flow = ConditionalCouplingFlow(self.padded_dim, cond_dim, layers=layers, hidden_dim=hidden_dim)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # z: (B, S, D)  cond: (B, S, cond_dim)
        bsz, slots, _ = z.shape
        flat_z = z.reshape(bsz * slots, self.slot_dim)
        flat_cond = cond.reshape(bsz * slots, -1)
        if self.padded_dim != self.slot_dim:
            flat_z = F.pad(flat_z, (0, self.padded_dim - self.slot_dim))
        out = self.flow(flat_z, flat_cond)
        if self.padded_dim != self.slot_dim:
            out = out[:, : self.slot_dim]
        return out.reshape(bsz, slots, self.slot_dim)


class FlowPMABlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = MemoryRead(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout, config=config)

        # PMA: learned seed/latent vectors that cross-attend to the token sequence.
        # Each layer has its own seeds (not weight-shared across layers).
        self.seeds = nn.Parameter(torch.randn(1, config.memory_slots, config.d_model) * 0.02)
        self.pma = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            batch_first=True,
        )
        self.cond_proj = nn.Linear(config.d_model, config.d_model)

        flow_layers = max(2, int(getattr(config, "flow_pma_layers", 2)))
        flow_hidden = max(128, config.d_model)
        self.slot_flow = _PerSlotFlow(
            slot_dim=config.d_model,
            cond_dim=config.d_model,
            hidden_dim=flow_hidden,
            layers=flow_layers,
        )

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        slots = self.config.memory_slots + (self.config.short_memory_slots if self.config.hierarchical_memory else 0)
        return x.new_zeros(x.shape[0], slots, self.config.d_model)

    def _update_memory(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        long_mem = memory[:, : self.config.memory_slots]
        seeds = self.seeds.expand(x.shape[0], -1, -1)
        # PMA cross-attention: seeds query token sequence to produce per-slot summaries.
        pooled, _ = self.pma(seeds, x, x, need_weights=False)
        # The pooled summary conditions a per-slot flow that transports the *existing*
        # slot toward its new attractor. Conditioning on both pooled summary and the
        # current slot would be circular through the residual; using only the pooled
        # summary keeps the flow's information source disjoint from the slot being moved.
        cond = self.cond_proj(pooled)
        transported = self.slot_flow(long_mem, cond)
        if not self.config.hierarchical_memory:
            return transported
        short = x[:, -self.config.short_memory_slots :]
        if short.shape[1] < self.config.short_memory_slots:
            short = F.pad(short, (0, 0, self.config.short_memory_slots - short.shape[1], 0))
        return torch.cat([transported, short], dim=1)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory is None:
            memory = self._initial_memory(x)
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))
        memory = self._update_memory(memory, x)
        return x, memory


class FlowPMALM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([FlowPMABlock(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, Any]:
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input length exceeds max_seq_len")
        x = self.token(input_ids)
        memory = None
        loops = max(1, getattr(self.config, "loop_count", 1))
        for _ in range(loops):
            for block in self.blocks:
                x, memory = block(x, memory)
        logits = self.head(self.ln(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        diagnostics = {
            "parameter_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "config": asdict(self.config),
        }
        return {"logits": logits, "loss": loss, "diagnostics": diagnostics}


def build_flow_pma_model(config: ModelConfig) -> nn.Module:
    return FlowPMALM(config)
