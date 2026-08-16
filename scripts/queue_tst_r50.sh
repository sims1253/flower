#!/usr/bin/env bash
# Rerun the one TST arm that crashed, LAST in the queue.
#
# It waits on the per-head confirmation rather than on the NorMuon re-screen,
# even though it is only ~35 min of work. The per-head waiter is already armed
# on the NorMuon summary.json, so gating this on the same file would fire both
# at once and put two training jobs on one 32 GB card — which on WSL2 does not
# OOM cleanly, it spills to host RAM and silently collapses throughput. Queueing
# behind the 11h confirmation costs nothing and cannot collide.
set -uo pipefail
cd "$(dirname "$0")/.."
export FLOWER_DATA_CACHE=data_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

TARGET=runs/sweep13_450m_longctx_fp8_perhead/summary.json
idle=0
[ -f runs/tst_r50_rerun/summary.json ] && { echo "[tst-r50] already complete"; exit 0; }

echo "[tst-r50] waiting for the per-head confirmation ($TARGET)"
while [ ! -f "$TARGET" ]; do
  if pgrep -f "flower.sweep" > /dev/null; then idle=0; else
    idle=$((idle+1)); [ "$idle" -ge 2 ] && { echo "[tst-r50] no sweep alive; proceeding"; break; }
  fi
  sleep 120
done
sleep 30
echo "[tst-r50] === TST bag2 r=0.5 rerun === $(date -Is)"
uv run python -m flower.sweep --config configs/tst_r50_rerun.yaml --output-dir runs/tst_r50_rerun
echo "[tst-r50] exit=$? at $(date -Is)"
