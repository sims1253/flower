"""Bloom-routed memory writes.

A neural analogue of a counting Bloom filter for memory addressing. The
mechanism is closest to Rae et al. 2019's Neural Bloom Filter, but adapted to
serve as a plug-in memory variant for a small autoregressive transformer with a
per-block summary -> memory write path (rather than the meta-learned external
key-value store from the original).

The write path each layer:

  1. Perceiver compression: a small set of learnable queries cross-attends into
     the local token window to produce P summary items (P = `bloom_summary_points`).
  2. K independent learnable "hash" projections, each summary item -> N_slots
     logits. A temperature-controlled softmax produces a soft top-k routing
     matrix per hash. Lower temperature = sharper hash, more aliasing risk.
  3. The K routing matrices are averaged (continuous superposition of OR'd bits)
     into one (B, P, N_slots) plan.
  4. A learned `write_value` projects each summary item; the plan distributes
     those values across slots additively (sparse, content-routed writes).

Why this matters for Flower: the existing variants either broadcast the same
update to every slot (summary_memory) or learn a single global summary->slot
mapping (perceiver style). Bloom routing gives each summary item its own
*structured* address that depends on content, with the "no false negatives"
property (every item has positive mass somewhere). Multiple items can collide
in the same slot; the read attention (`MemoryRead`) is asked to disentangle.
At small slot counts (16) this is a regulariser against over-specialisation
of any one slot; at larger slot counts (1024+) it gives real super-linear
capacity, which is the longer-term reason to want this in the toolbox.

Cost: K * d_model * N_slots params per layer per hash + P * d_model perceiver
queries. With defaults (K=4, P=16, S=16, D=384) that's ~120k extra params per
block -- negligible.
"""

from __future__ import annotations

import torch
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward
from flower.models.memory import MemoryRead


class BloomMemoryBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = MemoryRead(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout, config=config)

        # Perceiver compression to P summary items per chunk.
        self.summary_queries = nn.Parameter(
            torch.randn(1, config.bloom_summary_points, config.d_model) * 0.02
        )
        self.summary_attn = nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)

        # K independent learned hash projections. We initialise with small std so
        # different hashes diverge slowly and don't all collapse to the same
        # routing during the first few hundred steps.
        self.hashes = nn.ModuleList(
            [nn.Linear(config.d_model, config.memory_slots, bias=False) for _ in range(config.bloom_num_hashes)]
        )
        for h in self.hashes:
            nn.init.normal_(h.weight, std=0.05)

        # Value projection for the additive sparse write.
        self.write_value = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], self.config.memory_slots, self.config.d_model)

    def _bloom_route(self, items: torch.Tensor) -> torch.Tensor:
        """Average of K soft hash routings -> (B, P, N_slots) write plan."""
        temp = max(float(self.config.bloom_temperature), 1e-3)
        per_hash: list[torch.Tensor] = []
        for h in self.hashes:
            logits = h(items) / temp  # (B, P, S)
            per_hash.append(torch.softmax(logits, dim=-1))
        stacked = torch.stack(per_hash, dim=0)  # (K, B, P, S)
        plan = stacked.mean(dim=0)
        # Diagnostics: routing entropy (low = sharp Bloom-like routing, high =
        # softmax mush) and pairwise hash divergence (KL between heads; low =
        # hashes have collapsed onto the same routing -- K>1 buys nothing).
        with torch.no_grad():
            entropy = -(plan.clamp_min(1e-9) * plan.clamp_min(1e-9).log()).sum(dim=-1).mean()
            mean_routing = stacked.mean(dim=0, keepdim=True)
            kl = (stacked.clamp_min(1e-9) * (stacked.clamp_min(1e-9).log() - mean_routing.clamp_min(1e-9).log())).sum(dim=-1).mean()
            self.last_diag_bloom_routing_entropy = float(entropy.cpu())
            self.last_diag_bloom_hash_divergence = float(kl.cpu())
        return plan

    def _update_memory(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        queries = self.summary_queries.expand(bsz, -1, -1)
        items, _ = self.summary_attn(queries, x, x, need_weights=False)  # (B, P, D)
        plan = self._bloom_route(items)  # (B, P, S)
        values = self.write_value(items)  # (B, P, D)
        per_slot_write = plan.transpose(1, 2) @ values  # (B, S, D)
        return memory + per_slot_write / float(max(1, self.config.num_layers))

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory is None:
            memory = self._initial_memory(x)
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))
        memory = self._update_memory(memory, x)
        return x, memory


def build_bloom_memory_model(config: ModelConfig) -> CausalLM:
    blocks = [BloomMemoryBlock(config) for _ in range(config.num_layers)]
    return CausalLM(config, blocks)
