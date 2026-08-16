#!/usr/bin/env bash
# Queue the NorMuon re-screen behind the already-running queue
# (Muon screen -> TST screen -> this).
#
# WHY A SECOND SCRIPT RATHER THAN EDITING THE FIRST
#   bash reads a script incrementally from a byte offset while executing it.
#   Editing scripts/queue_after_confirmation.sh in place while it runs can make
#   the shell resume mid-token and execute garbage. Chaining a separate waiter
#   is the safe way to extend a live queue.
#
# WAIT CONDITION
#   The TST screen's summary.json, which flower.sweep writes only on success.
#   A bare file-wait would hang forever if that screen dies, so this also exits
#   the wait once no `flower.sweep` process has been alive for two consecutive
#   checks — then runs anyway, because a dead predecessor is a reason to take
#   the GPU, not to idle it.
#
# USAGE
#   nohup bash scripts/queue_normuon_fixed.sh > runs_queue_normuon.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

export FLOWER_DATA_CACHE=data_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

TARGET=runs/tst_screen_450m/summary.json
idle=0

echo "[normuon-queue] waiting for the TST screen ($TARGET)"
while [ ! -f "$TARGET" ]; do
  if pgrep -f "flower.sweep" > /dev/null; then
    idle=0
  else
    idle=$((idle + 1))
    if [ "$idle" -ge 2 ]; then
      echo "[normuon-queue] no flower.sweep alive across 2 checks; predecessor finished or died — proceeding"
      break
    fi
  fi
  sleep 120
done

sleep 30   # let the GPU release memory before the next allocation

echo "[normuon-queue] === NorMuon re-screen (4 arms x 1500 steps) === $(date -Is)"
uv run python -m flower.sweep \
  --config configs/muon_normuon_fixed_450m.yaml \
  --output-dir runs/muon_normuon_fixed_450m
echo "[normuon-queue] exit=$? at $(date -Is)"

echo "[normuon-queue] compare against the FIRST screen's broken NorMuon arms:"
echo "  runs/muon_screen_450m/muon_normuon.metrics.json          (+0.51682 bpb, BROKEN)"
echo "  runs/muon_normuon_fixed_450m/fixnorm_normuon.metrics.json (fixed)"
