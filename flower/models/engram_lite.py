from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward, count_parameters


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
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.local(self.ln1(x))
        x = x + self.engram(input_ids)
        x = x + self.ff(self.ln2(x))
        return x


class EngramLiteLM(CausalLM):
    def __init__(self, config: ModelConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([EngramLiteBlock(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, Any]:
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input length exceeds max_seq_len")
        x = self.token(input_ids)
        loops = max(1, getattr(self.config, "loop_count", 1))
        for _ in range(loops):
            for block in self.blocks:
                x = block(x, input_ids)
        logits = self.head(self.ln(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        diagnostics = {"parameter_count": count_parameters(self), "config": asdict(self.config)}
        return {"logits": logits, "loss": loss, "diagnostics": diagnostics}


def build_engram_lite_model(config: ModelConfig) -> CausalLM:
    return EngramLiteLM(config)
