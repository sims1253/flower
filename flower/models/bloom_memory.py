"""Bloom-routed memory writes.

A neural analogue of a counting Bloom filter for memory addressing. The
mechanism is closest to Rae et al. 2019's Neural Bloom Filter, but adapted to
serve as a plug-in memory variant for a small autoregressive transformer with a
per-block summary -> memory write path (rather than the meta-learned external
key-value store from the original).

The write path each layer:

  1. Perceiver compression: a small set of learnable queries cross-attends into
     the local token window to produce P summary items (P = `bloom_summary_points`).
  2. K independent learnable "hash" projections (stored as one `(K, D, S)`
     tensor, applied as a single batched matmul), each summary item -> N_slots
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
from flower.models.memory import MemoryRead, SDPCrossAttention


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
        # SDP cross-attention (compile-clean) replaces nn.MultiheadAttention: MHA
        # graph-breaks under torch.compile and OOMs at long context. Same params,
        # same cross-attention math (Q=summary queries, K=V=window, no causal
        # mask). See SDPCrossAttention docstring / NEXT_IDEAS.md section 4.
        self.summary_attn = SDPCrossAttention(config)

        # K independent learned hash projections, stored as a single (K, D, S)
        # tensor so all K projections are one batched matmul instead of a Python
        # loop over K nn.Linears (S14 Opportunity 2 Part A — K kernel launches
        # -> 1). We initialise with small std so different hashes diverge slowly
        # and don't all collapse to the same routing during the first few hundred
        # steps. std=0.05 matches the original `nn.init.normal_(h.weight, std=0.05)`.
        #
        # Layout note: each (D, S) slice is the *transpose* of the old
        # `nn.Linear(S, D).weight`, because einsum('bpd,kds->kbps') computes
        # items @ W_k.T  ==  the old h_k(items). remap_legacy_bloom_state_dict
        # below preserves this when loading old `hashes.{i}.weight` checkpoints.
        #
        # Optimizer-routing note (S14 Opp. 2 Part A constraint 3): the old
        # `hashes.{i}.weight` was 2D and matched no `memory_param_patterns`
        # entry, so it went to the Muon backbone group under the Muon optimizer.
        # This new Parameter is 3D `(K, D, S)`, and `_classify_params`
        # (optim.py) routes 2D -> Muon and everything else -> AdamW, so it now
        # lands in the AdamW backbone group instead. This changes the per-hash
        # update rule under `optimizer: muon` (sweep7/13 bloom runs). The
        # name-pattern routing is unchanged (still backbone, not memory);
        # adding `hash_weights` to `memory_param_patterns` is NOT what restores
        # Muon — that would require a 2D-per-hash split, which defeats Part A.
        # Flagged in the commit; revisit if a bloom Muon run regresses.
        self.hash_weights = nn.Parameter(
            torch.randn(config.bloom_num_hashes, config.d_model, config.memory_slots) * 0.05
        )

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
        # Single batched matmul: (B, P, D) @ (K, D, S) -> (K, B, P, S). Replaces
        # the K-length Python loop over nn.Linear (S14 Opportunity 2 Part A).
        logits = torch.einsum("bpd,kds->kbps", items, self.hash_weights) / temp
        stacked = logits.softmax(dim=-1)  # softmax over S for each (k,b,p); a
        # single softmax over the last dim of the (K,B,P,S) tensor is identical
        # to K·B·P independent softmaxes, so this matches the old per-hash loop
        # to float32 precision. Not bit-identical: the batched einsum and the K
        # separate matmuls pick different cuBLAS reduction orders, differing by
        # ~1 ULP (~1e-7 abs), well below training-relevant precision (validated
        # in tests/test_bloom_memory.py; see test docstring for the why).
        plan = stacked.mean(dim=0)
        # Diagnostics: routing entropy (low = sharp Bloom-like routing, high =
        # softmax mush) and pairwise hash divergence (KL between heads; low =
        # hashes have collapsed onto the same routing -- K>1 buys nothing).
        # Skipped under torch.compile: the host sync (float(...cpu())) graph-
        # breaks the compiled region and tanks GPU utilisation. The values are
        # only read by the diagnostic walker, which is itself disabled under
        # compile (collect_module_diagnostics), so dropping them there loses
        # nothing. (docs/training-speedups.md S14 Opportunity 4.)
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                # Hoist the clamped tensors once instead of calling .clamp_min
                # 5x; derive mean_routing from `plan` (already a mean over K)
                # via unsqueeze instead of recomputing stacked.mean().
                plan_safe = plan.clamp_min(1e-9)
                stacked_safe = stacked.clamp_min(1e-9)
                entropy = -(plan_safe * plan_safe.log()).sum(dim=-1).mean()
                log_plan = plan_safe.log().unsqueeze(0)  # (1,B,P,S) broadcast
                kl = (stacked_safe * (stacked_safe.log() - log_plan)).sum(dim=-1).mean()
                self.last_diag_bloom_routing_entropy = float(entropy.cpu())
                self.last_diag_bloom_hash_divergence = float(kl.cpu())
        return plan

    def _update_memory(self, memory: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        queries = self.summary_queries.expand(bsz, -1, -1)
        items = self.summary_attn(queries, x)  # (B, P, D)
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


def remap_legacy_bloom_state_dict(
    state_dict: dict[str, torch.Tensor],
    num_hashes: int | None = None,
) -> dict[str, torch.Tensor]:
    """Rewrite a pre-S14 bloom checkpoint's `hashes.{i}.weight` -> `hash_weights`.

    S14 Opportunity 2 Part A collapsed the K-hash `nn.ModuleList` into a single
    `(K, D, S)` `hash_weights` Parameter. An old checkpoint stores K separate
    `nn.Linear(S, D).weight` tensors per block; this stacks them (each
    transposed (D, S)) into the new single tensor so `load_state_dict` succeeds.

    Layout: each old `linear.weight` is `(S, D)` (out, in) and `linear(x)` =
    `x @ weight.T`; the new einsum `bpd,kds->kbps` needs the `(D, S)` slice, so
    each stacked slice is the old weight transposed.

    Idempotent: passes through a state_dict that is already in the new format.
    `num_hashes` (from `config.bloom_num_hashes`) is used only to size new
    tensors for blocks whose hashes are entirely absent (defensive); if None it
    is inferred from the largest hash index seen.
    """
    import re

    hash_re = re.compile(r"^(blocks\.\d+)\.hashes\.(\d+)\.weight$")
    # block_prefix -> {hash_idx: tensor}
    legacy: dict[str, dict[int, torch.Tensor]] = {}
    for key in list(state_dict.keys()):
        m = hash_re.match(key)
        if not m:
            continue
        prefix, idx = m.group(1), int(m.group(2))
        legacy.setdefault(prefix, {})[idx] = state_dict.pop(key)

    if not legacy:
        return state_dict  # already new-format (or not a bloom checkpoint)

    inferred_k = num_hashes
    if inferred_k is None:
        inferred_k = max((max(idxs) for idxs in legacy.values()), default=-1) + 1
        if inferred_k <= 0:
            return state_dict

    for prefix, idxs in legacy.items():
        # All present -> stack in order. Missing idx (shouldn't happen for a
        # real bloom checkpoint) would leave a zero slice, preserving key count.
        slices: list[torch.Tensor] = []
        for i in range(inferred_k):
            if i in idxs:
                w = idxs[i]  # (S, D)
                slices.append(w.t().contiguous())  # -> (D, S)
            else:
                # Defensive: shape from a sibling if any.
                ref = next(iter(idxs.values()))
                slices.append(torch.zeros(ref.shape[1], ref.shape[0], dtype=ref.dtype))
        state_dict[f"{prefix}.hash_weights"] = torch.stack(slices, dim=0)  # (K, D, S)
    return state_dict
