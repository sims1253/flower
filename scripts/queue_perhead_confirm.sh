#!/usr/bin/env bash
# Queue the per-head Muon 10k confirmation behind the running screens.
#
# A separate waiter rather than an edit to scripts/queue_resume.sh, which is
# live: bash reads a script incrementally from a byte offset while executing it,
# so editing it in place can make the shell resume mid-token.
#
# Waits for the NorMuon re-screen (the last item in the resumed queue). Falls
# through if no flower.sweep has been alive across two checks, so a dead
# predecessor releases the GPU instead of idling it. Skips its own work if
# already complete, so re-running is harmless.
#
# USAGE
#   nohup bash scripts/queue_perhead_confirm.sh > runs_queue_perhead.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

export FLOWER_DATA_CACHE=data_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

TARGET=runs/muon_normuon_fixed_450m/summary.json
OUT=runs/sweep13_450m_longctx_fp8_perhead
idle=0

if [ -f "$OUT/summary.json" ]; then
  echo "[perhead] already complete ($OUT/summary.json) — nothing to do."
  exit 0
fi

echo "[perhead] waiting for the NorMuon re-screen ($TARGET)"
while [ ! -f "$TARGET" ]; do
  if pgrep -f "flower.sweep" > /dev/null; then
    idle=0
  else
    idle=$((idle + 1))
    if [ "$idle" -ge 2 ]; then
      echo "[perhead] no flower.sweep alive across 2 checks — proceeding"
      break
    fi
  fi
  sleep 120
done

sleep 30

echo "[perhead] === per-head Muon 10k confirmation (~11h) === $(date -Is)"
uv run python -m flower.sweep \
  --config configs/sweep13_450m_longctx_fp8_perhead.yaml \
  --output-dir "$OUT"
echo "[perhead] exit=$? at $(date -Is)"

echo "[perhead] compare against the FP8 control:"
echo "  control  runs/sweep13_450m_longctx_fp8/vanilla_matched_fp8.metrics.json   val_bpb 0.90428"
echo "  this run $OUT/vanilla_matched_fp8_perhead.metrics.json"
echo "  criterion: at least 0.0004 BELOW the control (the converged 10k seed band)"
