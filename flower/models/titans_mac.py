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

Analytical fast path (``ModelConfig.titans_analytical_surprise=True``, S14
Opportunity 3 — research contribution): the inner surprise gradient has a
closed form, so we compute it directly with einsum + element-wise ops instead
of building/destroying an inner autograd graph every step. This is exact (it
matches ``torch.autograd.grad`` at fp32 ~1e-9) and the outer CE gradient still
flows through key_proj/val_proj/alpha_logit/write_scale, because every op in
the analytical path is a standard differentiable PyTorch op. See
``_surprise_analytical`` for the derivation and NEXT_IDEAS.md §7 for the
research framing.

This is structurally different from `summary_memory.py` (which uses max-pool +
MLP) and from the prior `titans_mac.py` stand-in (which used `|summary - memory|`
as a learned MLP gate, with no gradient signal at all).

Batch-size dependence of the write magnitude
(``ModelConfig.titans_per_sample_loss``): the inner associative-retrieval MSE
is by default reduced with a mean over BOTH the batch and feature dims (factor
2/(B*D); the legacy autograd path's ``F.mse_loss`` default is the same B*D
mean). The Titans write is ``alpha * write_scale * surprise``, and surprise is
the inner gradient, so every memory write scales with 1/B — memory at batch 1
runs ~B times "hotter" than at batch B. Training runs at
``training.batch_size`` while ``flower/eval.py``'s document-level paths
(``evaluate_documents``, ``sliding_window_document_loss``) score one document
at a time (B=1), so doc-level bpb was computed with batch_size-x larger memory
writes than training ever produced, and the two eval paths disagreed with each
other. ``titans_per_sample_loss=True`` switches BOTH surprise paths (analytical
and autograd) to a per-sample reduction — sum over batch, mean over D only
(factor 2/D) — making the memory dynamics batch-size-invariant and the B=1
document eval consistent with B=N training. ``causal_memory=True`` writes are
per-position and already per-row (factor 2/D per (batch, position) row), so
the flag is a no-op there. False (default) reproduces every published run
bit-for-bit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from flower.config import ModelConfig
from flower.models.base import CausalLM, CausalSelfAttention, FeedForward
from flower.models.memory import MemoryRead, causal_last_tokens


class TitansMACBlock(nn.Module):
    """Memory block whose write rule uses negative gradient of an inner KV loss.

    The write magnitude depends on the inner-loss reduction: legacy (default)
    reduces the inner MSE with a mean over batch AND dims, so memory writes
    scale with 1/batch_size; ``config.titans_per_sample_loss=True`` reduces
    per sample (sum over batch, mean over D) so the write dynamics are
    batch-size-invariant and the B=1 document eval matches B=N training. See
    the module docstring for the full story.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.d_model)
        self.local = CausalSelfAttention(config, config.local_window)
        self.ln_mem = nn.LayerNorm(config.d_model)
        self.mem_read = MemoryRead(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.ffn_dim, config.dropout, config=config)

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

    def _initial_memory_causal(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, S_total, D) per-position memory state for causal_memory=True."""
        slots = self.config.memory_slots + (self.config.short_memory_slots if self.config.hierarchical_memory else 0)
        return x.new_zeros(x.shape[0], x.shape[1], slots, self.config.d_model)

    def _inner_loss(self, memory: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Associative retrieval: read memory via key, MSE against target value.

        memory: (B, S, D) — only the long-mem prefix is involved.
        key:    (B, D)
        value:  (B, D)

        Attention weights are softmax over slots; the predicted value is the weighted
        average of slot contents. The loss is mean-squared error per element so the
        gradient magnitudes are stable across batch/dim.

        Reduction (``ModelConfig.titans_per_sample_loss``):
          False (legacy) -> mean over B and D (F.mse_loss default, factor
             2/(B*D)). The surprise — and therefore every memory write —
             scales with 1/B; batch-dependent by construction.
          True -> sum over B, mean over D only (factor 2/D). Each sample's
             inner problem contributes its own full MSE gradient, so the
             memory write for sample b does not shrink as the batch grows
             (batch-size-invariant dynamics).

        Used by the legacy autograd surprise path (when
        `titans_analytical_surprise=False`). The analytical path does not call
        this; it differentiates the same expression in closed form.
        """
        long_mem = memory[:, : self.config.memory_slots]
        scores = torch.einsum("bsd,bd->bs", long_mem, key) / (long_mem.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        predicted = torch.einsum("bs,bsd->bd", weights, long_mem)
        if self.config.titans_per_sample_loss:
            # Per-sample: sum over batch, mean over D. (sum over B*D) / D.
            return F.mse_loss(predicted, value, reduction="sum") / predicted.shape[-1]
        return F.mse_loss(predicted, value)

    def _surprise_analytical(
        self, long_mem: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Closed-form gradient of the inner MSE retrieval loss w.r.t. memory slots.

        Exact replacement for the legacy ``torch.autograd.grad(_inner_loss(...))``
        path. S14 Opportunity 3 (research contribution — analytical Titans
        surprise without autograd; see NEXT_IDEAS.md §7). Every op below is a
        standard differentiable PyTorch op, so the *outer* CE gradient still
        flows through key_proj / val_proj / alpha_logit / write_scale and through
        cross-layer memory — only the *inner* graph build/destroy is removed.

        Derivation (per batch element; ``D`` is d_model, ``B`` is batch):

            s_s   = <M_s, k> / sqrt(D)        score per slot
            w_s   = softmax(s)_s              weight per slot
            p     = sum_s w_s M_s             predicted value
            a     = p - v                     residual
            loss  = ||a||^2 / (B*D)           MSE (mean over B and D)

        Two gradient paths through M_s — direct (M_s in p) and through-softmax
        (M_s in s_s, which feeds every weight via the softmax Jacobian):

            d(loss)/d(M_s) = (2/(B*D)) * w_s * [ a + (<a, M_s - p>/sqrt(D)) * k ]

        The factor 2/(B*D) (not 2/D) comes from ``F.mse_loss``'s mean reduction
        over BOTH batch and feature dims; it must be kept to match the autograd
        path exactly and to preserve alpha_logit/write_scale semantics across
        checkpoints.

        With ``ModelConfig.titans_per_sample_loss=True`` the inner loss is
        instead reduced per sample — sum over batch, mean over D — so the
        factor becomes 2/D and the surprise (hence every memory write) is
        batch-size-invariant: memory writes at B=1 match those at B=N, which
        is what makes the B=1 document-level eval in flower/eval.py consistent
        with batch_size training. The closed form above is unchanged except
        for the scalar factor, so the analytical and autograd paths still
        match exactly with the flag on.

        Returns the NEGATIVE gradient (the Titans surprise signal,
        i.e. the direction that reduces the inner loss).

        Args:
            long_mem: (B, S, D) — the long-memory prefix.
            key:      (B, D)    — projected key.
            value:    (B, D)    — projected target value.

        Returns:
            (B, S, D) surprise = -d(loss)/d(M_s).
        """
        B, S, D = long_mem.shape
        sqrt_d = D ** 0.5
        scores = torch.einsum("bsd,bd->bs", long_mem, key) / sqrt_d       # (B,S)
        w = torch.softmax(scores, dim=-1)                                 # (B,S)
        p = torch.einsum("bs,bsd->bd", w, long_mem)                       # (B,D)
        a = p - value                                                     # (B,D)
        # <a, M_s - p> per slot — the softmax-Jacobian coupling term.
        dot = torch.einsum("bd,bsd->bs", a, long_mem - p.unsqueeze(1))    # (B,S)
        # Legacy: MSE mean reduction over B and D -> 2/(B*D) (batch-dependent
        # writes). Per-sample: sum over B, mean over D -> 2/D (invariant).
        factor = 2.0 / D if self.config.titans_per_sample_loss else 2.0 / (B * D)
        grad = factor * w.unsqueeze(-1) * (
            a.unsqueeze(1) + (dot / sqrt_d).unsqueeze(-1) * key.unsqueeze(1)
        )                                                                 # (B,S,D)
        return -grad  # negative gradient == Titans surprise

    def _surprise_update(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute the Titans surprise signal and apply it as a memory update."""
        summary = x.max(dim=1).values  # (B, D)
        key = self.key_proj(summary)
        value = self.val_proj(summary)

        long_mem = memory[:, : self.config.memory_slots]
        if self.config.titans_analytical_surprise:
            # Analytical path: closed-form inner gradient, no autograd graph.
            surprise = self._surprise_analytical(long_mem, key, value)
        else:
            # Legacy path: build an inner autograd graph and differentiate it
            # once per step. create_graph=True in training keeps the outer CE
            # gradient flowing through the surprise. enable_grad() is needed so
            # the inner graph is built even when the caller is under no_grad
            # (eval); the analytical path above does not need it.
            with torch.enable_grad():
                probe = long_mem.detach().requires_grad_(True)
                probe_memory = (
                    torch.cat([probe, memory[:, self.config.memory_slots :]], dim=1)
                    if memory.shape[1] > self.config.memory_slots
                    else probe
                )
                inner = self._inner_loss(probe_memory, key, value)
                (surprise_grad,) = torch.autograd.grad(
                    inner, probe, create_graph=self.training, retain_graph=True
                )
            surprise = -surprise_grad

        alpha = torch.sigmoid(self.alpha_logit).view(1, -1, 1)  # (1, S, 1)
        new_long = (1.0 - alpha) * long_mem + alpha * self.write_scale * surprise
        if not self.config.hierarchical_memory:
            return new_long
        short = x[:, -self.config.short_memory_slots :]
        if short.shape[1] < self.config.short_memory_slots:
            short = F.pad(short, (0, 0, self.config.short_memory_slots - short.shape[1], 0))
        return torch.cat([new_long, short], dim=1)

    def _surprise_update_causal(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """causal_memory=True write: memory is a per-position state (B, T, S_total, D).

        Same surprise rule and recurrence as ``_surprise_update``, but the
        summary at t is the PREFIX max (tokens <= t) and the retrieval probe
        at t reads/writes only the slot state at t. The per-position inner
        problems are independent, so the legacy (B, S, D) machinery is reused
        on the flattened (B*T) batch — including the autograd path, whose
        single torch.autograd.grad call covers all B*T probes at once.

        The inner loss is scaled by the row count so each position's surprise
        is the gradient of ITS OWN MSE (factor 2/D), not of a batch-wide mean
        (2/(B*T*D)). The batch-wide mean would make the memory at t depend on
        the number of positions in the window — a length leak that breaks
        prefix-truncation invariance.

        Interaction with ``titans_per_sample_loss``: the row-count rescale is
        only needed to cancel the legacy mean's 1/rows factor. With
        ``titans_per_sample_loss=True`` both surprise paths already reduce
        per row (factor 2/D each), so the rescale is skipped and the flag is
        exactly a no-op on this path — the causal write is per-row either way.
        """
        bsz, seq_len, _, dim = memory.shape
        rows = bsz * seq_len
        # 1 under per-sample reduction (already per-row), rows under the
        # legacy B-dependent reduction (cancels its 1/rows factor).
        per_row_scale = 1 if self.config.titans_per_sample_loss else rows
        summary = torch.cummax(x, dim=1).values  # (B, T, D)
        key = self.key_proj(summary).reshape(rows, dim)
        value = self.val_proj(summary).reshape(rows, dim)

        long_mem = memory[:, :, : self.config.memory_slots]
        s = self.config.memory_slots
        long_flat = long_mem.reshape(rows, s, dim)
        if self.config.titans_analytical_surprise:
            surprise = self._surprise_analytical(long_flat, key, value) * per_row_scale  # (B*T, S, D)
        else:
            with torch.enable_grad():
                probe = long_flat.detach().requires_grad_(True)
                if memory.shape[2] > s:
                    short_part = memory[:, :, s:]  # (B, T, short, D), detached with memory
                    probe_memory = torch.cat(
                        [probe.view(bsz, seq_len, s, dim), short_part], dim=2
                    ).reshape(rows, -1, dim)
                else:
                    probe_memory = probe
                inner = self._inner_loss(probe_memory, key, value) * per_row_scale
                (surprise_grad,) = torch.autograd.grad(
                    inner, probe, create_graph=self.training, retain_graph=True
                )
            surprise = -surprise_grad
        surprise = surprise.view(bsz, seq_len, s, dim)

        alpha = torch.sigmoid(self.alpha_logit).view(1, 1, -1, 1)  # (1, 1, S, 1)
        new_long = (1.0 - alpha) * long_mem + alpha * self.write_scale * surprise
        if not self.config.hierarchical_memory:
            return new_long
        short = causal_last_tokens(x, self.config.short_memory_slots)  # (B, T, n, D)
        return torch.cat([new_long, short], dim=2)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        causal = self.config.causal_memory
        if memory is None:
            memory = self._initial_memory_causal(x) if causal else self._initial_memory(x)
        x = x + self.local(self.ln1(x))
        x = x + self.mem_read(self.ln_mem(x), memory)
        x = x + self.ff(self.ln2(x))
        memory = self._surprise_update_causal(memory, x) if causal else self._surprise_update(memory, x)
        # Eval does not backprop: detach the memory state so it does not carry a
        # graph out of a no_grad context. (The analytical path has no inner
        # graph, so no enable_grad context is needed either way.)
        if not torch.is_grad_enabled():
            memory = memory.detach()
        return x, memory


def build_titans_mac_model(config: ModelConfig) -> CausalLM:
    blocks = [TitansMACBlock(config) for _ in range(config.num_layers)]
    return CausalLM(config, blocks)
