"""Titans MAC: Memory As Context with gradient-based surprise.

Reference: Behrouz et al., "Titans: Learning to Memorize at Test Time" (2024,
arXiv:2501.00663). The defining feature is the *surprise* signal used to gate
memory writes. In the original paper, surprise is the negative gradient of an
inner-loop memory loss with respect to the current memory state:

    s_t = -grad_M L_inner(M, x_t)

where L_inner is an associative-retrieval loss that asks whether the current memory
can reconstruct the incoming token's value from its key. Memory is updated as:

    M_{t+1} = (1 - alpha) * M_t + alpha * (s_t * write_value)

with alpha a learned per-slot decay/learning-rate.

This implementation gives Flower a dependency-free Titans variant where:
- The inner loss is reconstruction of the token-summary value from a learned key.
- The surprise is computed by torch.autograd.grad on that inner loss w.r.t. memory.
- The outer loss (cross-entropy on next token) backpropagates through the surprise
  via create_graph=True (Titans uses test-time updates, but for training we keep
  full differentiability so the memory-write rule itself is learned).

This is structurally different from `summary_memory.py` (which uses max-pool +
MLP) and from the prior `titans_mac.py` stand-in (which used `|summary - memory|`
as a learned MLP gate, with no gradient signal at all).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward
from flower.models.memory import MemoryRead


class TitansMACBlock(nn.Module):
    """Memory block whose write rule uses negative gradient of an inner KV loss."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = MemoryRead(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout)

        # Inner-loop probe: extract per-slot key/value pairs from the token summary.
        # The inner loss asks whether the memory's content (under a learned key) can
        # reconstruct the value. Surprise = -grad of this loss w.r.t. memory.
        self.key_proj = nn.Linear(config.d_model, config.d_model)
        self.val_proj = nn.Linear(config.d_model, config.d_model)
        self.query_proj = nn.Linear(config.d_model, config.d_model)

        # Per-slot learned forgetting rate (sigmoid -> [0,1]) and write scale.
        self.alpha_logit = nn.Parameter(torch.full((config.memory_slots,), -2.0))  # initially ~0.12
        self.write_scale = nn.Parameter(torch.tensor(1.0))

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        slots = self.config.memory_slots + (self.config.short_memory_slots if self.config.hierarchical_memory else 0)
        return x.new_zeros(x.shape[0], slots, self.config.d_model)

    def _inner_loss(self, memory: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Associative retrieval: read memory via key, MSE against target value.

        memory: (B, S, D) — only the long-mem prefix is involved.
        key:    (B, D)
        value:  (B, D)

        Attention weights are softmax over slots; the predicted value is the weighted
        average of slot contents. The loss is mean-squared error per element so the
        gradient magnitudes are stable across batch/dim.
        """
        long_mem = memory[:, : self.config.memory_slots]
        scores = torch.einsum("bsd,bd->bs", long_mem, key) / (long_mem.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        predicted = torch.einsum("bs,bsd->bd", weights, long_mem)
        return F.mse_loss(predicted, value)

    def _surprise_update(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute the Titans surprise signal and apply it as a memory update."""
        summary = x.max(dim=1).values  # (B, D)
        key = self.key_proj(summary)
        value = self.val_proj(summary)

        long_mem = memory[:, : self.config.memory_slots]
        # The surprise gradient must be taken w.r.t. the memory tensor with grad enabled.
        # We detach the input memory and request grad on a fresh leaf so the surprise
        # gradient is well-defined even if the input was produced under no_grad.
        probe = long_mem.detach().requires_grad_(True)
        # Inner loss uses the probe in place of `long_mem` inside _inner_loss. To keep
        # the inner-loss computation aligned with the slot-slice convention, we
        # construct a memory-shaped tensor with the probe in the long-mem prefix.
        probe_memory = (
            torch.cat([probe, memory[:, self.config.memory_slots :]], dim=1)
            if memory.shape[1] > self.config.memory_slots
            else probe
        )
        inner = self._inner_loss(probe_memory, key, value)
        # create_graph=True lets the outer cross-entropy backprop through the surprise.
        (surprise_grad,) = torch.autograd.grad(inner, probe, create_graph=self.training, retain_graph=True)
        surprise = -surprise_grad  # negative gradient = direction that reduces inner loss

        alpha = torch.sigmoid(self.alpha_logit).view(1, -1, 1)  # (1, S, 1)
        new_long = (1.0 - alpha) * long_mem + alpha * self.write_scale * surprise
        if not self.config.hierarchical_memory:
            return new_long
        short = x[:, -self.config.short_memory_slots :]
        if short.shape[1] < self.config.short_memory_slots:
            short = F.pad(short, (0, 0, self.config.short_memory_slots - short.shape[1], 0))
        return torch.cat([new_long, short], dim=1)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory is None:
            memory = self._initial_memory(x)
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))
        # Run surprise computation under grad even at eval time so the gradient is
        # available; the result is detached afterwards if we are in no_grad context.
        with torch.enable_grad():
            memory = self._surprise_update(memory, x)
        if not torch.is_grad_enabled():
            memory = memory.detach()
        return x, memory


def build_titans_mac_model(config: ModelConfig) -> CausalLM:
    blocks = [TitansMACBlock(config) for _ in range(config.num_layers)]
    return CausalLM(config, blocks)
