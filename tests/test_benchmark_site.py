import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "reports" / "benchmark_site"
# generate_data.py reads UNTRACKED run artifacts (runs/ and reports/ are
# gitignored). On a machine without the vast-pulled metrics — or in any fresh
# clone/checkout — the generated payload is legitimately empty and the
# content assertions below cannot hold. Skip rather than fail: this is machine
# state, not repo state (verified 2026-08-17: fails identically on pristine
# main when the pulled dir is absent).
SINGLE_GPU_SOURCE = ROOT / "runs" / "vast_status_36249420" / "pulled"


@pytest.mark.skipif(
    not SITE.exists() or not SINGLE_GPU_SOURCE.exists(),
    reason="benchmark site data needs the gitignored reports/ dir and untracked vast-pulled metrics",
)
def test_benchmark_site_data_generation():
    result = subprocess.run(
        ["python3", str(SITE / "generate_data.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "data.js" in result.stdout
    data_js = SITE / "data.js"
    assert data_js.exists()
    text = data_js.read_text()
    assert text.startswith("window.BENCHMARK_DATA = ")
    payload = json.loads(text.removeprefix("window.BENCHMARK_DATA = ").removesuffix(";\n"))
    assert payload["single_gpu"]
    assert payload["multi_gpu"]
    assert payload["bests"]["best_loss"]["variant"] == "summary_hierarchical_max"


@pytest.mark.skipif(
    not (SITE / "index.html").exists(),
    reason="benchmark site needs the gitignored reports/ dir (fresh clones/worktrees lack it)",
)
def test_benchmark_site_static_files_exist():
    assert (SITE / "index.html").exists()
    assert (SITE / "styles.css").exists()
    assert (SITE / "app.js").exists()
