from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.models.base import (
    CausalLM,
    CausalSelfAttention,
    FeedForward,
    _selective_checkpoint_context_fn,
    count_parameters,
)


class EngramResidual(nn.Module):
    """Causal hashed n-gram residual table.

    Each token position receives the sum of learned embeddings for the trailing
    n-grams ending at that position. Hashing keeps the memory fixed-size while
    letting the model condition on frequent adjacent token patterns.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.table_size = int(config.engram_table_size)
        self.ngram_min = int(config.engram_ngram_min)
        self.ngram_max = int(config.engram_ngram_max)
        if self.table_size <= 0:
            raise ValueError("engram_table_size must be positive")
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ValueError("engram_ngram_min/max must define a valid range")
        self.table = nn.Embedding(self.table_size, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.scale = nn.Parameter(torch.tensor(float(config.engram_scale)))
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def _hash_ngram(self, input_ids: torch.Tensor, n: int) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        padded = F.pad(input_ids, (n - 1, 0), value=0)
        hashed = torch.zeros(bsz, seq_len, dtype=torch.long, device=input_ids.device)
        # Fixed odd constants; modulo at each step avoids int64 overflow.
        base = 1_000_003
        for i in range(n):
            hashed = (hashed * base + padded[:, i : i + seq_len].long() + 1) % self.table_size
        return hashed

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        residual = 0.0
        for n in range(self.ngram_min, self.ngram_max + 1):
            residual = residual + self.table(self._hash_ngram(input_ids, n))
        return self.norm(residual) * self.scale


class EngramLiteBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.engram = EngramResidual(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout, config=config)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.local(self.ln1(x))
        x = x + self.engram(input_ids)
        x = x + self.ff(self.ln2(x))
        return x


class EngramLiteLM(CausalLM):
    """Engram-lite LM built on CausalLM's shared fields.

    Construction goes through ``CausalLM.__init__`` so every shared field the
    base owns (token/blocks/ln/tied head, ``fp8_lm_head``, ``bf16_cross_entropy``,
    ``fused_linear_ce``, ``activation_checkpoint``, MTP heads, attn_res
    plumbing, the ``_static_diagnostics`` cache and the init schemes) exists on
    this variant too. Previously the class hand-built the four shared modules
    and called ``nn.Module.__init__`` directly, so a sweep enabling any of
    those base flags silently did nothing here. Param counts and state_dict
    keys are unchanged for default configs (build_norm defaults to the same
    LayerNorm; MTP heads stay absent unless requested).

    The variant keeps only its extra: a forward that routes the raw
    ``input_ids`` into every block for the n-gram residual (CausalLM.forward
    threads a memory state through the blocks instead, which these blocks do
    not take). Everything the base flags own is routed through the parent
    helpers so the flags are live here too (pullfrog review of PR #11: the
    attributes existed but the forward bypassed the helpers, silently
    no-opping ``fp8_lm_head``/``bf16_cross_entropy``/``fused_linear_ce``/
    ``activation_checkpoint``):

    * final projection + loss go through ``CausalLM._compute_logits`` /
      ``_cross_entropy`` (FP8 eval head, BF16 CE);
    * the fused Liger CE path uses the same training+CUDA gates as base
      (``_ensure_liger_fce`` / ``_fused_cross_entropy``);
    * blocks are wrapped in ``torch.utils.checkpoint`` mirroring base.py's
      vanilla path ("ffn" falls back to full for these non-vanilla blocks,
      the documented base contract for memory-variant forwards).
    """

    def __init__(self, config: ModelConfig) -> None:
        # Blocks are constructed first because CausalLM.__init__ takes the
        # block list as an argument; from there the parent owns the shared
        # modules (token embedding, blocks, final norm, tied head).
        blocks = [EngramLiteBlock(config) for _ in range(config.num_layers)]
        super().__init__(config, blocks)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, Any]:
        # No max_seq_len cap: eval may run beyond the training length (Sweep 7
        # A1 relies on eval_seq_len > max_seq_len, and base CausalLM allows it
        # -- base.py:989). This is sound here because the n-gram hash is
        # position-independent (it hashes only the trailing n token ids, never
        # an absolute position), CausalSelfAttention builds its local causal
        # mask on-the-fly for any seq_len, and RotaryEmbedding extends its
        # cos/sin cache lazily. Nothing in this variant is tied to max_seq_len.
        x = self.token(input_ids)
        loops = max(1, getattr(self.config, "loop_count", 1))
        # S14-checkpoint: mirror CausalLM.forward's vanilla path — wrap each
        # block in torch.utils.checkpoint during training. use_reentrant=False
        # saves/restores RNG state (dropout identical across the forward and
        # the backward recompute) and accepts the (x, input_ids) signature
        # even though input_ids is a non-grad long tensor. Skipped in eval
        # (no backward -> checkpointing only wastes the recompute), when the
        # flag is off, and for loop_count > 1 (same documented limitation as
        # base.py). "ffn" has no clean FFN-only boundary here — the n-gram
        # residual interleaves with attention/FFN — so it falls back to FULL
        # checkpointing with a one-time notice, exactly the contract base.py
        # applies to non-TransformerBlock blocks.
        do_ckpt = (
            self.training
            and self.activation_checkpoint
            and loops == 1
        )
        if do_ckpt:
            from torch.utils.checkpoint import checkpoint

            if self.activation_checkpoint == "selective":
                # Selective: byte-threshold context_fn built once, reused
                # across layers (same as base).
                context_fn = _selective_checkpoint_context_fn()
                for block in self.blocks:
                    x = checkpoint(
                        block, x, input_ids,
                        use_reentrant=False,
                        context_fn=context_fn,
                    )
            else:
                if self.activation_checkpoint == "ffn" and not getattr(self, "_warned_ffn_fallback", False):
                    print(
                        "[checkpoint] activation_checkpoint='ffn' is only "
                        "supported on vanilla blocks; falling back to full "
                        "checkpointing for memory-variant blocks."
                    )
                    self._warned_ffn_fallback = True
                for block in self.blocks:
                    # True (full) and the "ffn" fallback both land here.
                    x = checkpoint(block, x, input_ids, use_reentrant=False)
        else:
            for _ in range(loops):
                for block in self.blocks:
                    x = block(x, input_ids)
        x_normed = self.ln(x)
        # S14-5b: fused lm_head + CE, same gates as CausalLM.forward —
        # training-time only, labels required, CUDA/Triton via
        # _ensure_liger_fce (which warns once and returns False off-CUDA, so
        # the eager path below runs instead — never a silent no-op).
        use_fused_ce = (
            self.fused_linear_ce
            and self.training
            and labels is not None
            and self._ensure_liger_fce()
        )
        if use_fused_ce:
            # Fused kernel projects the tied weight internally; logits are
            # never materialized. The weight is passed by reference so the
            # embedding gradient still flows through the tie.
            loss = self._fused_cross_entropy(x_normed, labels, self.token.weight, offset=1)
            logits = None
        else:
            # Helper routing (pullfrog): with every flag off these reduce to
            # exactly the previous inline ops — self.head(self.ln(x)) and the
            # inline shifted F.cross_entropy — so default numerics are
            # bit-identical (pinned by test_variant_small_fixes).
            logits = self._compute_logits(x_normed)
            loss = self._cross_entropy(logits, labels) if labels is not None else None
        diagnostics: dict[str, Any] = {}
        if self._static_diagnostics is None:
            self._static_diagnostics = {
                "parameter_count": count_parameters(self),
                "config": asdict(self.config),
            }
        diagnostics.update(self._static_diagnostics)
        return {"logits": logits, "loss": loss, "diagnostics": diagnostics}


def build_engram_lite_model(config: ModelConfig) -> CausalLM:
    return EngramLiteLM(config)
