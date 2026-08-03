from __future__ import annotations

import math

import torch
from torch import nn

from flower.config import ModelConfig


class MemoryRead(nn.Module):
    def __init__(self, config: ModelConfig, flow: nn.Module | None = None) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.kernel_bias = config.memory_kernel_bias
        self.q = nn.Linear(config.d_model, config.d_model)
        self.kv = nn.Linear(config.d_model, config.d_model * 2)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.flow = flow
        if self.kernel_bias not in {"none", "positional", "rbf"}:
            raise ValueError("memory_kernel_bias must be none, positional, or rbf")
        self.slot_bias = nn.Parameter(torch.zeros(config.memory_slots + config.short_memory_slots))
        self.rbf_scale = nn.Parameter(torch.tensor(1.0))
        # Sweep 7 (B2): log-sum-exp energy read. For small beta this behaves like
        # a mean read; as beta grows it sharpens toward max-energy retrieval.
        self.energy_read = getattr(config, "energy_read", False)
        if self.energy_read:
            beta_init = float(getattr(config, "energy_beta_init", 1.0))
            self.energy_log_beta = nn.Parameter(torch.tensor(math.log(max(beta_init, 1e-6))))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _bias(self, q_len: int, m_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.kernel_bias == "none":
            return None
        if self.kernel_bias == "positional":
            return self.slot_bias[:m_len].to(device=device, dtype=dtype).view(1, 1, 1, m_len)
        q_pos = torch.linspace(0, 1, q_len, device=device, dtype=dtype).view(q_len, 1)
        m_pos = torch.linspace(0, 1, m_len, device=device, dtype=dtype).view(1, m_len)
        scale = self.rbf_scale.abs().to(dtype=dtype) + 1e-6
        return (-(q_pos - m_pos).pow(2) * scale).view(1, 1, q_len, m_len)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self._split(self.q(x))
        k, v = self.kv(memory).chunk(2, dim=-1)
        k, v = self._split(k), self._split(v)
        if self.flow is not None:
            q = self.flow(q)
            k = self.flow(k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        bias = self._bias(x.shape[1], memory.shape[1], x.device, scores.dtype)
        if bias is not None:
            scores = scores + bias
        if self.energy_read:
            beta = self.energy_log_beta.exp().clamp_min(1e-6).to(device=scores.device)
            scores_f = scores.float()
            v_f = v.float()
            log_partition = torch.logsumexp(beta.float() * scores_f, dim=-1).unsqueeze(-1)
            out = (
                torch.logsumexp(beta.float() * (scores_f.unsqueeze(-1) + v_f.unsqueeze(2)), dim=-2)
                - log_partition
            ) / beta.float()
            out = out.to(dtype=v.dtype)
        else:
            attn = torch.softmax(scores, dim=-1)
            out = attn @ v
        out = out.transpose(1, 2).contiguous().view(x.shape)
        return self.out(out)


class MemoryMixinBlock(nn.Module):
    def initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def update_memory(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
