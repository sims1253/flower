"""Regression guard: Liger fused linear+CE must stay outside the Dynamo graph.

BACKGROUND
  `fused_linear_ce: true` (S14-5b) together with `compile_model: true` — the
  only configuration it was written for — crashed during the first compile on
  torch 2.13.0+cu130:

      InductorError: LoweringException: TypeError:
      tuned_addmm() takes 3 positional arguments but 4 were given
      target: aten.addmm.dtype

  Liger calls `torch.addmm(..., out_dtype=...)`, hitting the `aten.addmm.dtype`
  overload that inductor's `tuned_addmm` lowering does not accept. Reproduced
  both with and without FP8, so it is a plain Liger + torch.compile
  incompatibility on this build.

  The fix routes the call through `_call_liger_fce`, marked
  `@torch._dynamo.disable`. If that decorator is ever removed, compiled runs
  with fused CE start failing again several minutes into a run (during compile),
  which is an expensive way to find out — hence this cheap structural guard.
"""

from __future__ import annotations

import torch

from flower.models.base import _call_liger_fce


def test_liger_fce_call_is_dynamo_disabled():
    assert getattr(_call_liger_fce, "_torchdynamo_disable", False), (
        "_call_liger_fce lost its @torch._dynamo.disable decorator; compiling "
        "through Liger's aten.addmm.dtype raises InductorError (tuned_addmm "
        "takes 3 positional arguments but 4 were given)."
    )


def test_liger_fce_wrapper_forwards_the_call():
    """The wrapper must be a transparent pass-through, not reorder arguments.

    Liger's signature is (weight, input, labels) — a different order than the
    eager path's (input, weight). Getting it wrong would train a silently wrong
    loss rather than crash.
    """
    seen = {}

    def fake_fce(weight, inp, labels):
        seen["weight"] = weight
        seen["input"] = inp
        seen["labels"] = labels
        return torch.tensor(1.25)

    w = torch.zeros(8, 4)
    x = torch.ones(6, 4)
    y = torch.arange(6)
    out = _call_liger_fce(fake_fce, w, x, y)

    assert out.item() == 1.25
    assert seen["weight"] is w
    assert seen["input"] is x
    assert seen["labels"] is y


def test_dynamo_disable_survives_compilation():
    """Compiling a function that calls the wrapper must not trace into it."""
    traced = {"count": 0}

    def fake_fce(weight, inp, labels):
        traced["count"] += 1
        return (inp.sum() + weight.sum()) * 0.0 + labels.sum().float()

    def outer(w, x, y):
        return _call_liger_fce(fake_fce, w, x, y) * 2.0

    compiled = torch.compile(outer, dynamic=False)
    w = torch.zeros(4, 3)
    x = torch.ones(5, 3)
    y = torch.arange(5)

    result = compiled(w, x, y)
    expected = outer(w, x, y)
    assert torch.allclose(result, expected)
