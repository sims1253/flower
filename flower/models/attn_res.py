"""Depth-axis routing: Delta Block AttnRes with optional low-rank sliced keys.

Standard residual connections accumulate every sublayer's contribution into one
state, so a layer deep in the stack can only see the *sum* of what came before.
AttnRes (Kimi K3) applies the attention idea to the depth axis: each routing site
gets a learned query and attends over earlier depth representations.

Two published fixes to the original formulation are implemented here, because the
original degrades at scale (+6.9% ppl at 1044M, +6.6% at 7.6B):

* **Delta Block AttnRes** (arXiv:2605.18855). The original routes over cumulative
  states `s_i = h_i`; since each `h_i` is a running sum, adjacent sources share a
  growing common prefix, the softmax logits converge, and routing collapses
  toward uniform (max weight ~0.2 in deep layers). Routing over *deltas*
  `v_i = h_{i+1} - h_i` — what each block contributed rather than what
  accumulated — keeps sources structurally diverse (max weight ~0.6). Routing is
  **additive** (`h + sum a_i v_i`) rather than replacement (`sum a_i s_i`), so
  the residual stream survives.

* **S-LR-ATTNRES** (arXiv:2607.09694). In AttnRes each source plays two roles at
  once: the full-dimensional VALUE mixed into the stream, and the KEY used to
  score routing. Depth routing only discriminates among a handful of ordered
  sources, so the key needs far fewer dimensions than the model width. Slicing
  the key from the tail of the value costs no extra projection and no extra
  activation memory — the best loss/FLOPs Pareto point in the paper.

Both are exposed through `attn_res_key`: "full" scores against RMSNorm'd
full-width sources, "sliced" scores against `tail_r` of each source.

Deviation from the papers worth knowing: they rely on zero-init of the routing
query (uniform softmax) for a bounded perturbation at init. Uniform softmax over
deltas is still a non-zero shift, so this implementation additionally carries a
zero-initialised per-site output gate, making the module an *exact* identity at
initialisation. That keeps an AttnRes arm bit-comparable to its baseline at
step 0 and makes the mechanism safe to graft onto a pretrained base.
"""

from __future__ import annotations

import torch
from torch import nn

from flower.diag import should_collect, stash


class DepthRouter(nn.Module):
    """Routes over depth sources at `num_sites` block boundaries.

    Args:
        d_model: residual stream width.
        num_sites: number of routing sites (block boundaries).
        key_mode: "full" (score on RMSNorm'd d_model-wide sources) or "sliced"
            (score on the last `rank` dims of each source).
        rank: key width when `key_mode == "sliced"`.
    """

    def __init__(self, d_model: int, num_sites: int, key_mode: str = "full", rank: int = 64) -> None:
        super().__init__()
        if key_mode not in {"full", "sliced"}:
            raise ValueError(f"attn_res_key must be 'full' or 'sliced', got {key_mode!r}")
        if key_mode == "sliced" and not 0 < rank <= d_model:
            raise ValueError(f"attn_res_rank must be in (0, d_model={d_model}], got {rank}")
        self.d_model = d_model
        self.num_sites = num_sites
        self.key_mode = key_mode
        self.rank = rank if key_mode == "sliced" else d_model
        self.eps = 1e-6
        # Static per-site query. Zero-init -> uniform routing at step 0, which is
        # the paper's disruption-free starting point.
        self.query = nn.Parameter(torch.zeros(num_sites, self.rank))
        # Zero-init output gate -> exact identity at init (see module docstring).
        self.gate = nn.Parameter(torch.zeros(num_sites))

    def _keys(self, sources: torch.Tensor) -> torch.Tensor:
        """(B, T, S, d) -> (B, T, S, rank), RMS-normalised.

        RMSNorm on the key is what stops large-magnitude sources (typically the
        embedding and the earliest blocks) from dominating the routing scores.
        """
        k = sources if self.key_mode == "full" else sources[..., -self.rank :]
        rms = k.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (k.float() * rms).to(dtype=sources.dtype)

    def forward(self, site: int, hidden: torch.Tensor, sources: list[torch.Tensor]) -> torch.Tensor:
        """Additively mix routed depth sources into `hidden`.

        Args:
            site: index of this routing site (selects query/gate).
            hidden: (B, T, d) current residual stream.
            sources: list of (B, T, d) depth sources — the embedding followed by
                one delta per completed block.
        """
        values = torch.stack(sources, dim=2)  # (B, T, S, d)
        logits = (self._keys(values) * self.query[site]).sum(dim=-1)  # (B, T, S)
        weights = logits.softmax(dim=-1)
        mixed = (weights.unsqueeze(-1) * values).sum(dim=2)  # (B, T, d)
        # Routing-collapse diagnostic: the papers track mean max-softmax-weight
        # (AttnRes ~0.2 deep vs Delta ~0.6). Picked up by CausalLM's walker.
        # Stashed on-device (see flower/diag.py); the old float(...) here was a
        # per-step host sync and a graph break under compile.
        if should_collect():
            stash(self, "attn_res_max_weight", weights.max(dim=-1).values.mean())
        return hidden + self.gate[site] * mixed


def routing_sites(num_layers: int, num_blocks: int) -> list[int]:
    """Layer indices (0-based, inclusive) at which a routing site fires.

    Layers are partitioned into `num_blocks` contiguous groups; a site fires at
    the last layer of each group, and always at the final layer so no block
    delta is dropped. Requesting more blocks than layers degrades to one site
    per layer (full, rather than block, granularity).
    """
    if num_layers <= 0:
        return []
    per_block = max(1, -(-num_layers // max(1, num_blocks)))  # ceil division
    sites = [i for i in range(num_layers) if (i + 1) % per_block == 0]
    if not sites or sites[-1] != num_layers - 1:
        sites.append(num_layers - 1)
    return sites
