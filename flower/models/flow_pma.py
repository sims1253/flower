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

import math
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.flows.coupling import ConditionalCouplingFlow
from flower.models.base import CausalSelfAttention, FeedForward
from flower.models.memory import MemoryRead, causal_chunked_map, causal_last_tokens, causal_prefix_attention


class _PerSlotFlow(nn.Module):
    """Conditional coupling flow applied independently to each slot.

    The slot dimension D may be odd; if so we pad to even, run the flow, and unpad.

    Accepts z of shape (B, S, D) (legacy) or (B, T, S, D) (causal_memory=True
    per-position states): everything but the last dim is folded into the flow's
    batch, so the same parameters serve both paths.
    """

    def __init__(self, slot_dim: int, cond_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.padded_dim = slot_dim + (slot_dim % 2)
        self.flow = ConditionalCouplingFlow(self.padded_dim, cond_dim, layers=layers, hidden_dim=hidden_dim)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # z: (B, S, D) or (B, T, S, D); cond matches z's leading dims.
        rows = z.numel() // self.slot_dim
        flat_z = z.reshape(rows, self.slot_dim)
        flat_cond = cond.reshape(rows, -1)
        if self.padded_dim != self.slot_dim:
            flat_z = F.pad(flat_z, (0, self.padded_dim - self.slot_dim))
        out = self.flow(flat_z, flat_cond)
        if self.padded_dim != self.slot_dim:
            out = out[:, : self.slot_dim]
        return out.reshape(z.shape)


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

    def _initial_memory_causal(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, S_total, D) per-position memory state for causal_memory=True."""
        slots = self.config.memory_slots + (self.config.short_memory_slots if self.config.hierarchical_memory else 0)
        return x.new_zeros(x.shape[0], x.shape[1], slots, self.config.d_model)

    def _causal_pma(self, x: torch.Tensor) -> torch.Tensor:
        """Per-position prefix PMA summaries using ``self.pma``'s MHA weights.

        Same parameters as the whole-window ``self.pma(seeds, x, x)`` call
        (in_proj / out_proj, no new params) and the same math (per-head
        scaling 1/sqrt(head_dim)), but every position t's pooled summaries
        attend to x[:, :t+1] only. Returns (B, T, S, D) where S is the seed
        count (= memory_slots). Mirrors flow_ot_memory._causal_source_attn.
        """
        bsz, seq_len, d_model = x.shape
        heads = self.pma.num_heads
        head_dim = d_model // heads
        w_q, w_k, w_v = self.pma.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.pma.in_proj_bias.chunk(3, dim=0)
        seeds = self.seeds.expand(bsz, -1, -1)
        q = F.linear(seeds, w_q, b_q).view(bsz, -1, heads, head_dim).permute(0, 2, 1, 3)  # (B, H, S, hd)
        k = F.linear(x, w_k, b_k).view(bsz, seq_len, heads, head_dim).permute(0, 2, 1, 3)  # (B, H, T, hd)
        v = F.linear(x, w_v, b_v).view(bsz, seq_len, heads, head_dim).permute(0, 2, 1, 3)
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)  # (B, H, S, T)
        ctx = causal_prefix_attention(scores, v)  # (B, H, T, S, hd)
        ctx = ctx.permute(0, 2, 3, 1, 4).reshape(bsz, seq_len, -1, d_model)  # (B, T, S, D)
        return self.pma.out_proj(ctx)

    def _update_memory(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.config.causal_memory:
            return self._update_memory_causal(memory, x)
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

    def _update_memory_causal(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """causal_memory=True write: memory is a per-position state (B, T, S_total, D).

        Same seed cross-attention -> conditioning projection -> per-slot flow
        pipeline as ``_update_memory`` (identical parameters), but the pooled
        summary at t attends to tokens <= t only and the flow transports the
        slot state at t. The hierarchical short-memory slice becomes the last
        n tokens of the prefix ending at t (left-padded for early positions),
        exactly like the other causal short-memory writes. The per-slot flow
        runs chunked over T to bound its intermediates (causal_chunked_map).
        """
        long_mem = memory[:, :, : self.config.memory_slots]  # (B, T, S, D)
        pooled = self._causal_pma(x)  # (B, T, S, D)
        cond = self.cond_proj(pooled)
        transported = causal_chunked_map(self.slot_flow, long_mem, cond)
        if not self.config.hierarchical_memory:
            return transported
        short = causal_last_tokens(x, self.config.short_memory_slots)  # (B, T, n, D)
        return torch.cat([transported, short], dim=2)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory is None:
            memory = self._initial_memory_causal(x) if self.config.causal_memory else self._initial_memory(x)
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
