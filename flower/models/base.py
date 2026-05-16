from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def causal_mask(seq_len: int, device: torch.device, local_window: int | None = None) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    mask = idx[:, None] >= idx[None, :]
    if local_window is not None:
        mask &= (idx[:, None] - idx[None, :]) < local_window
    return mask


def scaled_dot_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, local_window: int | None = None) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.local_window = local_window
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)
        self.out = nn.Linear(config.d_model, config.d_model)
        # S2.4: optional RBF distance kernel as additive attention bias.
        # rbf bias = -scale * (i-j)^2 / window^2, learnable scale, per-head.
        self.kernel_bias = getattr(config, "local_attn_kernel_bias", "none")
        if self.kernel_bias not in {"none", "rbf"}:
            raise ValueError("local_attn_kernel_bias must be 'none' or 'rbf'")
        if self.kernel_bias == "rbf":
            init_scale = float(getattr(config, "local_attn_rbf_scale", 4.0))
            self.rbf_log_scale = nn.Parameter(torch.full((self.num_heads,), math.log(max(init_scale, 1e-3))))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _rbf_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Normalised positions so behaviour is window-size invariant.
        denom = float(self.local_window) if self.local_window is not None else float(seq_len)
        idx = torch.arange(seq_len, device=device, dtype=dtype) / max(denom, 1.0)
        d2 = (idx[:, None] - idx[None, :]).pow(2)  # (T, T)
        scale = self.rbf_log_scale.exp().to(device=device, dtype=dtype).view(self.num_heads, 1, 1)
        return (-scale * d2).unsqueeze(0)  # (1, H, T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        mask = causal_mask(x.shape[1], x.device, self.local_window).view(1, 1, x.shape[1], x.shape[1])
        if self.kernel_bias == "rbf":
            scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
            scores = scores + self._rbf_bias(x.shape[1], x.device, scores.dtype)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            out = attn @ v
        else:
            out = scaled_dot_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(x.shape)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, attention: nn.Module | None = None) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = attention or CausalSelfAttention(config, config.local_window)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, memory


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, Any]:
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input length exceeds max_seq_len")
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token(input_ids) + self.pos(pos).unsqueeze(0)
        memory = None
        diagnostics: dict[str, Any] = {}
        # A2 looped transformer: run the block stack `loop_count` times. Memory state
        # carries across loops, so the bank acts as a scratchpad / persistent state.
        # loop_count=1 reproduces the standard single-pass behaviour.
        loops = max(1, getattr(self.config, "loop_count", 1))
        for _ in range(loops):
            for block in self.blocks:
                x, memory = block(x, memory)
        logits = self.head(self.ln(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        diagnostics["parameter_count"] = count_parameters(self)
        diagnostics["config"] = asdict(self.config)
        return {"logits": logits, "loss": loss, "diagnostics": diagnostics}
