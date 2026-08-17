"""Tests for the `last_diag_*` diagnostic stash pattern (flower/diag.py).

The invariant under test, from the pullfrog review of the cleanup branch: a
`last_diag_*` value that a forward did NOT refresh must never be reported as
current. Before the fix, every migrated site only wrote the stash when
`should_collect()` was True, so under torch.compile the attribute froze at
the last eager forward's value (warmup, a validation pass on the same
instance) — or, for MeanFlow, the diagnostics dict fabricated a 0.0.
"""

from __future__ import annotations

import pytest
import torch

from flower.config import ModelConfig
from flower.diag import clear, stash
from flower.flows.hamiltonian import HamiltonianFlow, WalnutsHamiltonianFlow
from flower.models import build_model
from flower.models.attn_res import DepthRouter


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


def _simulate_compile(monkeypatch):
    """Make `should_collect()` False without a real Dynamo trace.

    Dynamo folds `torch.compiler.is_compiling()` to True while tracing, so
    this reproduces exactly the branch the compiled graph takes, while the
    module itself keeps running eagerly (which is also what an eval pass on
    the eager copy of a compiled model sees).
    """
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)


def _tiny_cfg(**overrides) -> ModelConfig:
    base = dict(
        variant="vanilla_local",
        vocab_size=128,
        d_model=32,
        num_heads=4,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=16,
        local_window=4,
        memory_slots=4,
        flow_steps=1,
    )
    base.update(overrides)
    return ModelConfig(**base)


# --------------------------------------------------------------------------
# The helpers themselves
# --------------------------------------------------------------------------


def test_stash_then_clear_roundtrip():
    m = torch.nn.Linear(4, 4)
    t = torch.randn(2, 4).mean()
    stash(m, "foo", t)
    assert isinstance(m.last_diag_foo, torch.Tensor)
    assert m.last_diag_foo.ndim == 0
    assert not m.last_diag_foo.requires_grad
    # The walker (flower/models/base.py) skips non-tensor/non-float values,
    # so a cleared stash is invisible to it — not reported as zero.
    clear(m, "foo")
    assert m.last_diag_foo is None
    # A later eager forward overwrites the None, no re-registration needed.
    stash(m, "foo", t)
    assert isinstance(m.last_diag_foo, torch.Tensor)


# --------------------------------------------------------------------------
# MeanFlow: the diagnostics dict must not emit stale values or a fake 0.0
# --------------------------------------------------------------------------


def test_meanflow_aux_loss_not_stale_or_fabricated_under_compile(monkeypatch):
    model = build_model(_tiny_cfg(variant="flow_meanflow"))
    tokens = torch.randint(0, 128, (2, 16))

    # No eager forward ran yet: under compile the key must be absent (None),
    # not a fabricated 0.0 (train.py's logging skips non-float values).
    _simulate_compile(monkeypatch)
    out = model(tokens, labels=tokens)
    assert out["diagnostics"]["meanflow_aux_loss"] is None
    monkeypatch.undo()

    # Eager forward stashes the real value...
    out = model(tokens, labels=tokens)
    eager = out["diagnostics"]["meanflow_aux_loss"]
    assert isinstance(eager, torch.Tensor) and eager.ndim == 0

    # ...and a compiled forward must clear it rather than re-report it.
    _simulate_compile(monkeypatch)
    out = model(tokens, labels=tokens)
    assert out["diagnostics"]["meanflow_aux_loss"] is None
    monkeypatch.undo()

    # Eager again: the diagnostic comes back (clearing is not permanent).
    out = model(tokens, labels=tokens)
    assert isinstance(out["diagnostics"]["meanflow_aux_loss"], torch.Tensor)


# --------------------------------------------------------------------------
# AttnRes: the walker must not aggregate a stale stash
# --------------------------------------------------------------------------


def test_attn_res_walker_omits_stale_stash_under_compile(monkeypatch):
    model = build_model(_tiny_cfg(attn_res="delta_block", attn_res_blocks=1))
    assert model.depth_router is not None
    tokens = torch.randint(0, 128, (2, 16))

    out = model(tokens, labels=tokens)
    assert "attn_res_max_weight_mean" in out["diagnostics"]
    assert model.depth_router.last_diag_attn_res_max_weight is not None

    _simulate_compile(monkeypatch)
    out = model(tokens, labels=tokens)
    assert model.depth_router.last_diag_attn_res_max_weight is None
    # The walker (which still runs here — collect_module_diagnostics is an
    # independent flag) must not aggregate the stale value as current.
    assert "attn_res_max_weight_mean" not in out["diagnostics"]


# --------------------------------------------------------------------------
# Hamiltonian flows: stash cleared under compile, restored on eager re-entry
# --------------------------------------------------------------------------


def test_hamiltonian_flows_clear_stale_stash_under_compile(monkeypatch):
    x = torch.randn(2, 8)
    keys = (
        "last_diag_hamiltonian_energy_drift",
        "last_diag_hamiltonian_substeps_mean",
        "last_diag_hamiltonian_substeps_max",
    )
    for flow in (HamiltonianFlow(8, steps=2), WalnutsHamiltonianFlow(8, steps=2)):
        flow(x)
        drift = flow.last_diag_hamiltonian_energy_drift
        assert isinstance(drift, torch.Tensor) and drift.ndim == 0

        _simulate_compile(monkeypatch)
        flow(x)
        assert flow.last_diag_hamiltonian_energy_drift is None
        if isinstance(flow, WalnutsHamiltonianFlow):
            assert flow.last_diag_hamiltonian_substeps_mean is None
            assert flow.last_diag_hamiltonian_substeps_max is None
        monkeypatch.undo()

        # Eager re-entry (train.py keeps calling the eager model for
        # validation) must re-stash, not crash on a cleared slot.
        flow(x)
        assert isinstance(flow.last_diag_hamiltonian_energy_drift, torch.Tensor)


def test_hamiltonian_attention_walker_omits_stale_stash_under_compile(monkeypatch):
    model = build_model(_tiny_cfg(variant="hamiltonian_attention")).eval()
    tokens = torch.randint(0, 128, (2, 16))

    with torch.no_grad():
        out = model(tokens, labels=tokens)
    assert "hamiltonian_energy_drift_mean" in out["diagnostics"]

    _simulate_compile(monkeypatch)
    with torch.no_grad():
        out = model(tokens, labels=tokens)
    assert "hamiltonian_energy_drift_mean" not in out["diagnostics"]


# --------------------------------------------------------------------------
# Real Dynamo: the clearing branch is what gets traced, so it must not
# graph-break (a plain attribute store, no host sync).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module,args",
    [
        (HamiltonianFlow(8, steps=2), (torch.randn(2, 8),)),
        (DepthRouter(16, 2), (0, torch.randn(2, 4, 16), [torch.randn(2, 4, 16)])),
    ],
)
def test_clear_branch_traces_without_graph_breaks(module, args):
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(module)(*args)
    assert explanation.graph_break_count == 0


def test_compiled_forward_clears_and_eager_restores():
    torch._dynamo.reset()
    flow = HamiltonianFlow(8, steps=2)
    x = torch.randn(2, 8)
    flow(x)  # eager warmup stashes a value
    assert flow.last_diag_hamiltonian_energy_drift is not None

    compiled = torch.compile(flow)
    for _ in range(2):
        compiled(x)
        assert flow.last_diag_hamiltonian_energy_drift is None

    flow(x)  # eager re-entry
    assert flow.last_diag_hamiltonian_energy_drift is not None
