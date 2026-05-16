#!/usr/bin/env bash
set -euo pipefail

uv run ruff format --check .
uv run ruff check .
uv run ty check --warn all --output-format concise
uv run pytest -q
