"""Surprise-gated memory: cheap LM-loss-style write gating without 2nd-order grads.

Motivation: Titans MAC (titans_mac.py) gates memory writes by the *gradient* of an
inner-loop associative-retrieval loss. That signal is theoretically principled but
expensive (it requires `torch.autograd.grad(..., create_graph=True)` inside every
block, costing a full extra backward through the inner loss every layer-step) and
unstable at small scales (the gradient magnitudes depend strongly on init).

This variant tests the same hypothesis -- "surprise = memorable" -- with a much
cheaper signal. A tiny per-layer "judge" net emits a per-token scalar `s_t` from
the local hidden state. That scalar plays two roles in the memory write:

  - Per-token weighting in the summary: high-surprise tokens dominate the
    aggregate that gets written into memory (so the slots learn to store the
    *interesting* parts of the chunk, not its average).
  - Global write gate: sigmoid(mean(s_t) - threshold) multiplies the candidate
    update so confident / predictable chunks barely touch memory at all.

The judge has no auxiliary loss. The LM cross-entropy is the only training
signal: gradient flows from the next-token loss back through the gate, through
the judge, and rewards judge outputs that produce useful (loss-reducing) writes.
The judge therefore learns "what is *worth* remembering for next-token prediction"
end-to-end, without ever computing a 2nd-order gradient or an inner loss.

This is structurally cheaper than Titans MAC and conceptually closer to the
"the LM's own confusion drives memorability" interpretation than the inner-loop
formulation.
"""

from __future__ import annotations

import torch
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward
from flower.models.memory import MemoryRead, causal_prefix_attention, causal_running_mean


class SurpriseMemoryBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = MemoryRead(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout, config=config)

        # Judge: x -> per-token surprise scalar.
        self.judge = nn.Sequential(
            nn.Linear(config.d_model, config.surprise_judge_dim),
            nn.GELU(),
            nn.Linear(config.surprise_judge_dim, 1),
        )
        # Learnable global threshold; init at 0 so gate ~ 0.5 at the start and
        # writes are neither fully on nor off until the judge has shaped itself.
        self.surprise_threshold = nn.Parameter(torch.zeros(1))

        # Standard summary memory write components (kept simple to isolate the
        # surprise-gating effect from other architectural confounds).
        self.token_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.mem_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.update = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], self.config.memory_slots, self.config.d_model)

    def _initial_memory_causal(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, S, D) per-position memory state for causal_memory=True."""
        return x.new_zeros(x.shape[0], x.shape[1], self.config.memory_slots, self.config.d_model)

    def _update_memory(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.config.causal_memory:
            return self._update_memory_causal(memory, x)
        # Per-token surprise score from the judge.
        surprise = self.judge(x).squeeze(-1)  # (B, T)
        # Softmax-weighted summary: peaks at the most "surprising" tokens.
        weights = torch.softmax(surprise * self.config.surprise_scale, dim=-1)  # (B, T)
        token_summary = (weights.unsqueeze(-1) * x).sum(dim=1, keepdim=True)  # (B, 1, D)
        token_summary = token_summary.expand(-1, self.config.memory_slots, -1)

        combined = self.token_mlp(token_summary) + self.mem_mlp(memory)
        candidate = self.update(combined) / float(max(1, self.config.num_layers))

        # Global write gate (one scalar per batch element).
        gate = torch.sigmoid(surprise.mean(dim=-1, keepdim=True) - self.surprise_threshold)
        self._emit_diagnostics(gate, surprise)
        return memory + gate.unsqueeze(-1) * candidate

    def _update_memory_causal(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """causal_memory=True write: memory is a per-position state (B, T, S, D).

        Same judge/gate/MLPs as the legacy write (no new parameters), but the
        surprise-weighted summary at t softmaxes over tokens <= t only and the
        gate at t is the sigmoid of the PREFIX-mean surprise.
        """
        # Per-token surprise score from the judge.
        surprise = self.judge(x).squeeze(-1)  # (B, T)
        # Prefix softmax: position t's summary weights tokens <= t only. The
        # judge scores are the (single-row) attention logits over tokens.
        scores = (surprise * self.config.surprise_scale).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
        token_summary = causal_prefix_attention(scores, x.unsqueeze(1))  # (B, 1, T, 1, D)
        token_summary = token_summary.squeeze(1).squeeze(2).unsqueeze(2)  # (B, T, 1, D)
        token_summary = token_summary.expand(-1, -1, self.config.memory_slots, -1)

        combined = self.token_mlp(token_summary) + self.mem_mlp(memory)
        candidate = self.update(combined) / float(max(1, self.config.num_layers))

        # Global write gate, per position: sigmoid of the prefix-mean surprise.
        gate = torch.sigmoid(causal_running_mean(surprise) - self.surprise_threshold)  # (B, T)
        self._emit_diagnostics(gate, surprise)
        return memory + gate.unsqueeze(-1).unsqueeze(-1) * candidate

    def _emit_diagnostics(self, gate: torch.Tensor, surprise: torch.Tensor) -> None:
        # Diagnostics: track gate magnitude (is the judge actually firing?) and
        # surprise score spread (has the judge learned to discriminate, or
        # collapsed to a constant?).
        # Skipped under torch.compile: the host syncs (float(...cpu())) graph-
        # break the compiled region (same guard/pattern as bloom_memory).
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                self.last_diag_surprise_gate_mean = float(gate.mean().detach().cpu())
                self.last_diag_surprise_score_std = float(surprise.std().detach().cpu())

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory is None:
            memory = self._initial_memory_causal(x) if self.config.causal_memory else self._initial_memory(x)
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))
        memory = self._update_memory(memory, x)
        return x, memory


def build_surprise_memory_model(config: ModelConfig) -> CausalLM:
    blocks = [SurpriseMemoryBlock(config) for _ in range(config.num_layers)]
    return CausalLM(config, blocks)
