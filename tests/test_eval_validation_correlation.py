"""Tests for scripts/eval_validation_correlation.py.

Pins the NaN-handling contract from the pullfrog review of PR #3:
`memory_ablation_probe` reports NaN deltas plus `ablated: False` for variants
whose memory read cannot be patched. The aggregation must exclude those rows
from the ablation mean and from the rank agreement — a NaN through
`statistics.mean` propagates, and through `sorted()` silently scrambles the
ranking verdict. (The probe-side `ablated`/NaN contract itself is pinned in
tests/test_eval_probe_correctness.py.)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "eval_validation_correlation", ROOT / "scripts" / "eval_validation_correlation.py"
)
assert spec is not None and spec.loader is not None
evc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evc)

# needle_bp per variant, chosen so a scrambled (NaN-keyed) sort would produce a
# visibly different order than the true one.
NEEDLE = {"vanilla_local": 1.0, "summary_memory": 2.0, "phase_memory": 3.0}


def _make_fake_run(root: Path, name: str) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "model_step10.pt").touch()


def _stub(monkeypatch, root: Path, patchable: set[str]) -> None:
    for variant in ("vanilla_local", "summary_memory", "phase_memory"):
        for seed in (0, 1):
            _make_fake_run(root, f"{variant}_seed{seed}")

    def fake_run_one(ckpt: Path, device, doc_limit: int) -> dict:
        variant = evc._variant_of(ckpt.parent.name)
        ablated = variant in patchable
        return {
            "text_recall_bp": 16.0,
            "mqar_bp": 8.0,
            "needle_bp": NEEDLE[variant],
            "needle_curve": {},
            "ablation_delta_bpb": (0.05 if ablated else float("nan")),
            "ablated": ablated,
        }

    monkeypatch.setattr(evc, "run_one", fake_run_one)
    monkeypatch.setattr(evc, "resolve_device", lambda _: "cpu")


def test_unablated_variants_reported_not_corrupted(tmp_path, monkeypatch, capsys):
    # Only summary_memory is patchable: vanilla has no memory, phase_memory's
    # read path is outside the patchable set (the exact case the review flagged).
    _stub(monkeypatch, tmp_path, patchable={"summary_memory"})
    evc.main(["--root", str(tmp_path)])
    out = capsys.readouterr().out

    assert "unmeasured" in out
    # No NaN leaks into the printed means (careful: "vanilla" contains "nan").
    assert "delta=nan" not in out.lower() and "±nan" not in out.lower()
    # One measurable variant cannot produce a ranking: skip explicitly rather
    # than print a NaN-scrambled or bogus verdict.
    assert "rank agreement skipped" in out


def test_rank_agreement_uses_measurable_subset(tmp_path, monkeypatch, capsys):
    _stub(monkeypatch, tmp_path, patchable={"summary_memory", "phase_memory"})
    evc.main(["--root", str(tmp_path)])
    out = capsys.readouterr().out

    assert ("rankings agree" in out) or ("rankings DIFFER" in out)
    for line in out.splitlines():
        if "ranking (low->high)" in line:
            # The unmeasurable vanilla rows must not sit inside the rankings.
            assert "vanilla_local" not in line
            assert "summary_memory" in line and "phase_memory" in line
