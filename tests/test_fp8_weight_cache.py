"""Tests for `fp8_weight_cache`.

The cache measured a NET LOSS at this model's shape (0.973x) and is not wired
in — see its docstring. These tests exist anyway because the two properties they
guard are the ones that would make it dangerous if anyone ever enables it:

  1. it must be bit-identical to the uncached path (it is a pure reuse of an
     identical computation, so anything else is a bug), and
  2. it must FAIL LOUD if a weight changes while the cache is live.

Property 2 is not paranoia. The natural implementation — invalidate on
`tensor._version` — is broken here, because `p.data -= x` does not bump the
version counter while `p.add_(x)` does, and `flower/optim.py`'s Muon uses both.
A version-based cache would serve stale quantized weights after cautious weight
decay and train silently worse.

CUDA-only: torchao's FP8 path requires it.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _fp8_model(d: int = 256, hid: int = 512) -> nn.Module:
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    from torchao.float8.config import Float8LinearRecipeName

    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(d, hid, bias=False), nn.Linear(hid, d, bias=False))
    m = m.cuda().to(torch.bfloat16)
    convert_to_float8_training(
        m, config=Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.TENSORWISE)
    )
    return m


def test_version_counter_would_be_an_unsafe_invalidation_signal():
    """Documents *why* this is a context manager and not auto-invalidation.

    If a future torch makes `p.data -= x` bump `_version`, this test fails and
    the safer-but-clunkier design can be revisited.
    """
    p = nn.Parameter(torch.randn(4, 4))
    before = p._version
    with torch.no_grad():
        p.data -= 0.1  # the pattern Muon's cautious weight decay uses
    assert p._version == before, "p.data -= now bumps _version; revisit the cache design"
    with torch.no_grad():
        p.add_(torch.ones(4, 4))
    assert p._version > before, "p.add_ should bump _version"


@cuda_only
def test_cached_is_bit_identical_to_uncached():
    from flower.precision import fp8_weight_cache

    accum = 3
    xs = [torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) for _ in range(accum)]

    def run(use_cache: bool):
        m = _fp8_model()
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            if use_cache:
                with fp8_weight_cache(m):
                    for x in xs:
                        m(x).square().mean().backward()
            else:
                for x in xs:
                    m(x).square().mean().backward()
            opt.step()
        return [p.detach().float().clone() for p in m.parameters()]

    for a, b in zip(run(False), run(True), strict=True):
        assert torch.equal(a, b), "cached path diverged from uncached"


@cuda_only
def test_raises_when_a_weight_changes_while_cache_is_live():
    """The silent-staleness failure must become a loud one."""
    from flower.precision import fp8_weight_cache

    m = _fp8_model()
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="stale"):
        with fp8_weight_cache(m):
            m(x).sum().backward()
            with torch.no_grad():
                # Exactly the mutation `_version` does not catch.
                m[0].weight.data -= 0.1
            m(x).sum().backward()


@cuda_only
def test_forwards_are_restored_after_exit():
    """A leaked patched forward would keep serving a stale weight forever."""
    from flower.precision import fp8_weight_cache

    m = _fp8_model()
    before = [mod.forward for mod in m]
    with fp8_weight_cache(m):
        pass
    for mod, orig in zip(m, before, strict=True):
        assert mod.forward == orig, "forward was not restored on exit"


@cuda_only
def test_exception_inside_cache_still_restores_forwards():
    from flower.precision import fp8_weight_cache

    m = _fp8_model()
    before = [mod.forward for mod in m]
    with pytest.raises(ValueError):
        with fp8_weight_cache(m):
            raise ValueError("boom")
    for mod, orig in zip(m, before, strict=True):
        assert mod.forward == orig, "forward leaked after an exception"
