#!/usr/bin/env bash
# Resume the screen queue after it was held for GPU time.
#
# The original queue was two chained wrapper scripts; both were killed to hold
# the GPU, which stopped new jobs from launching while leaving the in-flight
# sweep alone (killing a wrapper does not signal its already-running child).
# This restarts whatever is still outstanding.
#
# It is IDEMPOTENT: each screen is skipped if its summary.json already exists,
# so running this twice will not redo finished work, and it is safe to run
# without first checking what completed.
#
# It also refuses to start while another sweep is alive, so an accidental launch
# cannot put two training jobs on one 32 GB card — which on WSL2 does not OOM
# cleanly but spills to host RAM and silently collapses throughput.
#
# USAGE
#   nohup bash scripts/queue_resume.sh > runs_queue_resume.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

export FLOWER_DATA_CACHE=data_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

if pgrep -f "flower.sweep" > /dev/null; then
  echo "[resume] a flower.sweep is already running; refusing to start a second."
  echo "[resume] wait for it, then re-run this script."
  exit 1
fi

run_screen () {  # name, config, outdir
  local name=$1 config=$2 outdir=$3
  if [ -f "$outdir/summary.json" ]; then
    echo "[resume] SKIP $name — already complete ($outdir/summary.json)"
    return
  fi
  echo "[resume] === $name === $(date -Is)"
  uv run python -m flower.sweep --config "$config" --output-dir "$outdir"
  echo "[resume] $name exit=$? at $(date -Is)"
  sleep 30
}

run_screen "Muon screen"      configs/muon_screen_450m.yaml         runs/muon_screen_450m
run_screen "TST screen"       configs/tst_screen_450m.yaml          runs/tst_screen_450m
run_screen "NorMuon re-screen" configs/muon_normuon_fixed_450m.yaml runs/muon_normuon_fixed_450m

echo "[resume] all outstanding work finished at $(date -Is)"
