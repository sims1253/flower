"""Shared helper for the `last_diag_*` diagnostic stash pattern.

CausalLM's diagnostics walker (flower/models/base.py) aggregates any
`last_diag_<field>` attribute across submodules and emits `<field>_mean` /
`<field>_max` (0-d tensors included — the walker does the single host
transfer at logging time). Modules that want to feed it must follow two
rules, both of which are easy to get wrong inline (several sites did):

1. Never sync inside forward. `float(t)` / `t.cpu()` / `t.item()` on a CUDA
   tensor blocks the host until the device catches up, serialising a
   pipeline that train.py otherwise keeps fully asynchronous.
2. Never compute the value under torch.compile. A host sync graph-breaks
   the compiled region, and the walker itself is disabled under compile
   (`collect_module_diagnostics = False`, see train.py), so the value would
   be computed-but-never-read — pure cost.

`should_collect()` is the single guard for both rules; `stash()` is the
write side. The pattern was established in bloom_memory._bloom_route; this
module exists so new sites copy one honest helper instead of a comment.
"""

from __future__ import annotations

import torch
from torch import nn


def should_collect() -> bool:
    """True when per-forward diagnostic values are worth computing.

    False while Dynamo is tracing (torch.compile): the walker that reads
    these values is disabled under compile, so computing them would only
    add a graph break.
    """
    return not torch.compiler.is_compiling()


def stash(module: nn.Module, key: str, tensor: torch.Tensor) -> None:
    """Record a diagnostic scalar on `module.last_diag_<key>`.

    `tensor` must already be 0-d — reduce at the call site (`.mean()`,
    `.max()`, ...) — because the walker only picks up 0-d values. It is
    stored detached and on-device: no `.cpu()`/`float()` here, ever. The
    walker does the one host transfer, once, at logging time.

    A 0-d result of a reduction allocates its own one-element storage, so
    stashing keeps no reference to the activation it was reduced from.
    """
    setattr(module, f"last_diag_{key}", tensor.detach())
