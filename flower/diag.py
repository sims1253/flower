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
write side and `clear()` the invalidate side — a forward that does not
refresh the value (the not-collecting branch) must clear it, or the walker
will happily emit the last eager value as if it were current. The stash
pattern was established in bloom_memory._bloom_route; this module exists so
new sites copy one honest helper instead of a comment.
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


def clear(module: nn.Module, key: str) -> None:
    """Drop the stashed `last_diag_<key>` so it cannot be read as current.

    The counterpart to `stash` for forwards that did NOT refresh the value:
    call this in the `else` of the `should_collect()` guard. Under
    torch.compile the stash branch is skipped forever, but the attribute set
    by an earlier eager forward (warmup, a validation pass on the same
    instance) survives — and the walker would emit that stale number as if
    it came from the current step. Setting `None` (rather than deleting)
    also pins "never collected" uniformly: the walker skips non-tensor,
    non-float values, so nothing stale or fabricated is reported. A later
    eager forward simply overwrites it via `stash` again.
    """
    setattr(module, f"last_diag_{key}", None)
