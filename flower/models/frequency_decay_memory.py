"""Frequency-aware memory decay -- inverse spaced repetition.

Standard memory architectures (and human-inspired ones like Titans) apply a
uniform or per-slot static decay rate. The user's intuition for this variant
goes the other way: information that recurs frequently in the local attention
window is already retrievable cheaply, so the *long-term* memory should
preferentially preserve content that is *rare but distinctive*.

Mechanism: alongside the memory tensor we carry a per-slot running magnitude
of writes ("write_mag", shape (B, S)). Each layer:

  1. Compute a candidate update like summary_memory (max-pool aggregate, MLP).
  2. Per-slot the *effective* decay is:
       decay_t = base_decay * (1 + freq_penalty * write_mag_t)
     so slots that have been hit hard already this forward pass get decayed
     more aggressively, freeing them to absorb the next signal.
  3. Apply: memory <- (1 - decay_t) * memory + candidate.
  4. Update running magnitude: write_mag <- write_mag + ||candidate||_2 detached
     (we don't backprop through the magnitude tracker -- it's a heuristic gate,
     not a learnable parameter).

This is novel as far as I can tell: existing "forgetting" mechanisms (Titans,
recurrent gates, Mamba's selective state) all rely on *learned* per-slot or
per-token decay, but none of them adapt decay to the slot's recent *traffic*
across layer depth. The Bloom-filter literature has no temporal analogue
either -- they use static hash functions.

Practical scope: write_mag resets per forward pass (per sequence). It is not
a long-term frequency estimator across training; it captures only the
within-sequence pressure on each slot. That keeps the mechanism local and
gradient-safe. Note that within a single forward the accumulator sums over
every block (and every loop when loop_count > 1), which is what drives the
decay saturation documented at the clamp in FrequencyDecayBlock._update.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward, count_parameters
from flower.models.memory import MemoryRead


class FrequencyDecayBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = MemoryRead(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout, config=config)

        # Standard summary memory write components.
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

        # Per-slot base decay rate; sigmoid into [0, 1]. Init at config default
        # (default -2 -> ~0.12) so most slots start with mild forgetting.
        self.decay_logit = nn.Parameter(
            torch.full((config.memory_slots,), float(config.freq_decay_init))
        )

    def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=1, keepdim=True).values

    def _update(
        self,
        memory: torch.Tensor,
        write_mag: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        causal = self.config.causal_memory
        # Causal form: prefix max at t, so the pooled token the candidate is
        # built from at t sees tokens <= t only.
        token_summary = torch.cummax(x, dim=1).values if causal else self._aggregate(x)
        if causal:
            token_summary = token_summary.unsqueeze(2).expand(-1, -1, self.config.memory_slots, -1)  # (B, T, S, D)
        else:
            token_summary = token_summary.expand(-1, self.config.memory_slots, -1)  # (B, S, D)
        combined = self.token_mlp(token_summary) + self.mem_mlp(memory)
        candidate = self.update(combined) / float(max(1, self.config.num_layers))

        base = torch.sigmoid(self.decay_logit).view(1, -1)  # (1, S)
        # Heavier traffic -> stronger decay. Clamp so the multiplier stays in
        # [base, 1] (cap at full decay) and we don't overshoot into negatives.
        # write_mag: (B, S) legacy / (B, T, S) causal — the per-slot traffic
        # accumulated over layers at this position's own state.
        #
        # SATURATION CONSEQUENCE (do not change without a flag — this is the
        # documented behaviour): write_mag is a running SUM over every write
        # this forward has made to the slot (each block adds its candidate's
        # detached L2 norm, and with loop_count > 1 the loops keep adding on
        # top with no reset). Once base * (1 + freq_penalty * write_mag) >= 1
        # the clamp pins effective decay at 1.0: retention (1 - effective)
        # becomes 0 and new_memory == candidate — the slot's previous content
        # is fully discarded on that write, and stays fully discarded for the
        # rest of the forward (the accumulator only grows within a pass).
        effective = base * (1.0 + self.config.freq_penalty * write_mag)
        effective = torch.clamp(effective, max=1.0)  # (B, S) / (B, T, S)

        new_memory = (1.0 - effective).unsqueeze(-1) * memory + candidate
        # Detach: write_mag is a heuristic accumulator, not a learnable signal.
        slot_mag = candidate.detach().norm(dim=-1)  # (B, S) / (B, T, S)
        return new_memory, write_mag + slot_mag

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        causal = self.config.causal_memory
        if state is None:
            if causal:
                memory = x.new_zeros(x.shape[0], x.shape[1], self.config.memory_slots, self.config.d_model)
                write_mag = x.new_zeros(x.shape[0], x.shape[1], self.config.memory_slots)
            else:
                memory = x.new_zeros(x.shape[0], self.config.memory_slots, self.config.d_model)
                write_mag = x.new_zeros(x.shape[0], self.config.memory_slots)
        else:
            memory, write_mag = state
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))
        memory, write_mag = self._update(memory, write_mag, x)
        # Per-block write-traffic diagnostic (the base.py `last_diag_` walker
        # convention): mean of the accumulated write_mag at this block's exit.
        # Host sync — skipped under torch.compile, same guard pattern as the
        # LM-level diagnostics (PR #5).
        if not torch.compiler.is_compiling():
            self.last_diag_write_mag = float(write_mag.mean().detach().cpu())
        return x, (memory, write_mag)


class FrequencyDecayLM(CausalLM):
    """Custom LM that threads (memory, write_mag) state across blocks.

    Construction goes through ``CausalLM.__init__`` so every shared field the
    base owns (token/blocks/ln/tied head, ``fp8_lm_head``, ``bf16_cross_entropy``,
    ``fused_linear_ce``, ``activation_checkpoint``, MTP heads, attn_res
    plumbing, the ``_static_diagnostics`` cache and the init schemes) exists on
    this variant too. Previously the class subclassed ``nn.Module`` directly
    and hand-built the four shared modules, so a sweep enabling any of those
    base flags silently did nothing here. Param counts and state_dict keys are
    unchanged for default configs (build_norm defaults to the same LayerNorm;
    MTP heads stay absent unless requested).

    The variant keeps only its extra: a forward that threads the
    (memory, write_mag) tuple through the blocks and surfaces the write-mag
    diagnostics (CausalLM.forward threads a single memory tensor instead).
    """

    def __init__(self, config: ModelConfig) -> None:
        # Blocks are constructed first because CausalLM.__init__ takes the
        # block list as an argument; from there the parent owns the shared
        # modules (token embedding, blocks, final norm, tied head).
        blocks = [FrequencyDecayBlock(config) for _ in range(config.num_layers)]
        super().__init__(config, blocks)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, Any]:
        # No max_seq_len cap: eval may run beyond the training length (Sweep 7
        # A1 relies on eval_seq_len > max_seq_len, and base CausalLM allows it
        # -- base.py:989). This is sound here: the memory and write_mag states
        # are allocated per forward from the input's own T (legacy (B, S, D) /
        # (B, S); causal (B, T, S, D) / (B, T, S)), so nothing is tied to
        # max_seq_len; CausalSelfAttention builds its local causal mask
        # on-the-fly for any seq_len and RotaryEmbedding extends its cos/sin
        # cache lazily.
        x = self.token(input_ids)
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        loops = max(1, getattr(self.config, "loop_count", 1))
        # write_mag accumulates over EVERY block of EVERY loop within this
        # forward (state is only None at the very start; it is never reset
        # between blocks or loops). Each block adds its per-slot candidate
        # magnitudes on top, so by deep blocks / later loops the accumulated
        # magnitude can push `base * (1 + freq_penalty * write_mag)` past the
        # clamp in FrequencyDecayBlock._update — see the saturation note there.
        for _ in range(loops):
            for block in self.blocks:
                x, state = block(x, state)
        logits = self.head(self.ln(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        if self._static_diagnostics is None:
            self._static_diagnostics = {
                "parameter_count": count_parameters(self),
                "config": asdict(self.config),
            }
        diagnostics: dict[str, Any] = dict(self._static_diagnostics)
        # Surface the write-magnitude distribution as a diagnostic so the
        # composite eval can correlate freq-decay activity with downstream metrics.
        # Skipped under torch.compile: the host syncs (float(...cpu())) graph-
        # break the compiled region (same guard/pattern as bloom_memory).
        if state is not None and not torch.compiler.is_compiling():
            _, write_mag = state
            diagnostics["frequency_decay_mean_mag"] = float(write_mag.mean().detach().cpu())
            diagnostics["frequency_decay_max_mag"] = float(write_mag.max().detach().cpu())
            # Per-block view of the same accumulator (each block stashes its
            # exit value via the base.py `last_diag_` convention). The
            # final-state numbers above only show the endpoint; these show how
            # the traffic built up across the stack, which is what drives the
            # saturation described in FrequencyDecayBlock._update.
            block_mags = [
                float(getattr(block, "last_diag_write_mag"))
                for block in self.blocks
                if getattr(block, "last_diag_write_mag", None) is not None
            ]
            if block_mags:
                diagnostics["write_mag_mean"] = sum(block_mags) / len(block_mags)
                diagnostics["write_mag_max"] = max(block_mags)
        return {"logits": logits, "loss": loss, "diagnostics": diagnostics}


def build_frequency_decay_memory_model(config: ModelConfig) -> CausalLM:
    return FrequencyDecayLM(config)
