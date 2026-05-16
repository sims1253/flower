"""Pre-download FineWeb-Edu sample-10BT parquet shards once for offline reuse.

Set FLOWER_DATA_CACHE before running (e.g. /workspace/hf_data_cache). Downloads
the full ~30GB sample-10BT dataset; subsequent training runs read parquet files
locally instead of byte-range streaming over HTTP. This eliminates the
mid-run httpx-client-closed crashes seen in Sweep 2.

Idempotent: re-running skips files already present (snapshot_download dedups).

Usage (locally or on instance):
  FLOWER_DATA_CACHE=/workspace/hf_data_cache uv run python scripts/prefetch_dataset.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    cache_dir = os.environ.get("FLOWER_DATA_CACHE")
    if not cache_dir:
        print("ERROR: set FLOWER_DATA_CACHE to a path with ~50GB free", file=sys.stderr)
        sys.exit(2)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    print(f"Downloading FineWeb-Edu sample-10BT parquet shards to {cache_path}")
    print("  (skipping any already-present shards; this is idempotent)")

    snapshot_download(
        repo_id="HuggingFaceFW/fineweb-edu",
        repo_type="dataset",
        allow_patterns=["sample/10BT/*.parquet"],
        local_dir=str(cache_path),
        max_workers=8,
    )

    parquets = sorted(cache_path.glob("sample/10BT/*.parquet"))
    total_bytes = sum(p.stat().st_size for p in parquets)
    print(f"Done. {len(parquets)} parquet shards, {total_bytes / 1e9:.1f} GB on disk.")


if __name__ == "__main__":
    main()
