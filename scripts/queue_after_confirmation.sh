#!/usr/bin/env bash
# Queue the Muon and TST screens behind the running FP8 confirmation run.
#
# WHY A QUEUE AND NOT JUST LAUNCHING
#   Sharing the GPU is not free. While benchmarking alongside the confirmation
#   run its throughput fell from ~58.9k to 57.1k tok/s (~3%), and a heavier
#   overlap risks pushing peak VRAM past the 32 GB card into WSL2's silent
#   host-RAM spill, where throughput collapses by an order of magnitude with no
#   error. So: one job at a time.
#
# WHAT IT WAITS FOR
#   runs/sweep13_450m_longctx_fp8/summary.json, written by flower.sweep only
#   after the variant completes. Polling a file rather than a PID survives this
#   script being restarted.
#
# USAGE
#   nohup bash scripts/queue_after_confirmation.sh > runs_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

export FLOWER_DATA_CACHE=data_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

CONFIRM_DONE=runs/sweep13_450m_longctx_fp8/summary.json

echo "[queue] waiting for confirmation run: $CONFIRM_DONE"
while [ ! -f "$CONFIRM_DONE" ]; do sleep 120; done
echo "[queue] confirmation run finished at $(date -Is)"

# Give the GPU a moment to release memory before the next allocation.
sleep 30

echo "[queue] === Muon variant screen (10 arms x 1500 steps) ==="
uv run python -m flower.sweep \
  --config configs/muon_screen_450m.yaml \
  --output-dir runs/muon_screen_450m
echo "[queue] Muon screen exit=$? at $(date -Is)"

sleep 30

echo "[queue] === TST screen (4 arms x 1500 steps) ==="
uv run python -m flower.sweep \
  --config configs/tst_screen_450m.yaml \
  --output-dir runs/tst_screen_450m
echo "[queue] TST screen exit=$? at $(date -Is)"

echo "[queue] all queued work finished at $(date -Is)"
echo "[queue] analyse with:"
echo "  PYTHONPATH=. uv run python scripts/analyze_speedup_screen.py \\"
echo "      --run-dir runs/muon_screen_450m --control muon_baseline \\"
echo "      --seed-dir runs/speedup_screen_450m_seed1"
