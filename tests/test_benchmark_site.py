import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "reports" / "benchmark_site"


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


def test_benchmark_site_static_files_exist():
    assert (SITE / "index.html").exists()
    assert (SITE / "styles.css").exists()
    assert (SITE / "app.js").exists()
