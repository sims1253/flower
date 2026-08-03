"""Hyperparameter transfer rules.

Implements the Complete(d)P scaling rules (Apple Research, arXiv:2512.22382) for
moving a *measured* learning rate between model widths, depths, batch sizes, and
training durations. The point is to calibrate once, empirically, then transfer —
rather than re-sweeping at every configuration or carrying forward a constant
whose provenance nobody remembers.

The rules are multipliers on a reference LR:

| dimension | multiplier            | why |
|-----------|-----------------------|-----|
| width     | `1 / m_N`             | muP: update size must not grow with width |
| depth     | `m_L ** (alpha - 1)`  | depth-muP; alpha=0.5 is the standard choice |
| batch     | `sqrt(m_B)`           | larger batch -> less gradient noise -> bigger step |
| duration  | `1 / sqrt(kappa)`     | the token-horizon correction |

The **duration** rule is the one most often missed and the one that bit Delphi:
the optimal LR *decreases* as a run gets longer, so an LR tuned on a short probe
is systematically too high for the real run. A 2500-step calibration transferred
to a 15000-step run needs a factor of `1/sqrt(6)` ~ 0.41. Ignoring this is how
short-horizon sweeps produce settings that diverge at full length.

These are approximations, not laws. Use them to place a sweep grid and to
transfer between nearby configurations; do not use them to skip measuring
entirely, especially across an architecture change.
"""

from __future__ import annotations

from dataclasses import dataclass

DEPTH_ALPHA = 0.5


@dataclass(frozen=True)
class RunShape:
    """The axes an LR transfers along."""

    width: int  # d_model
    depth: int  # num_layers
    batch_tokens: int  # effective batch size in tokens (batch * accum * seq_len)
    steps: int

    @property
    def total_tokens(self) -> int:
        return self.batch_tokens * self.steps


def transfer_lr(lr: float, source: RunShape, target: RunShape, depth_alpha: float = DEPTH_ALPHA) -> float:
    """Scale a learning rate measured on `source` to `target`.

    >>> ref = RunShape(width=768, depth=14, batch_tokens=65536, steps=2500)
    >>> full = RunShape(width=768, depth=14, batch_tokens=65536, steps=15000)
    >>> round(transfer_lr(0.01, ref, full), 5)   # duration correction only
    0.00408
    """
    if min(source.width, source.depth, source.batch_tokens, source.steps) <= 0:
        raise ValueError("source RunShape fields must all be positive")
    if min(target.width, target.depth, target.batch_tokens, target.steps) <= 0:
        raise ValueError("target RunShape fields must all be positive")

    m_width = target.width / source.width
    m_depth = target.depth / source.depth
    m_batch = target.batch_tokens / source.batch_tokens
    kappa = target.total_tokens / source.total_tokens

    return lr * (1.0 / m_width) * (m_depth ** (depth_alpha - 1.0)) * (m_batch**0.5) * (kappa**-0.5)


def horizon_correction(source_steps: int, target_steps: int) -> float:
    """Duration-only multiplier, for transferring a short probe to a long run.

    Separated out because it is the correction that gets forgotten: the other
    three multipliers are 1.0 whenever the probe uses the real model and batch,
    which is the usual calibration setup.
    """
    if source_steps <= 0 or target_steps <= 0:
        raise ValueError("step counts must be positive")
    return (target_steps / source_steps) ** -0.5
