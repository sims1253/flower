#!/usr/bin/env bash
# Window follow-up, chained behind the MTP screen.
#
# Gated on runs/mtp_screen_450m/summary.json — the last item of
# scripts/queue_screens.sh. Separate script for the same reason queue_screens.sh
# is separate: the running queue holds an open fd on its own (now-deleted) inode
# and will not pick up edits, and restarting it mid-screen would discard hours.
#
# See configs/window_followup_450m.yaml for what this answers and why. Short
# version: the window screen measured -0.062 val_bpb at equal wall-clock, which
# is 2.2x the best optimizer result on this model, but (a) the optimum is
# unbracketed below 512 and (b) it may be a 1500-step warmup artifact. This
# brackets the curve and re-runs the headline comparison at ~3x the horizon.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

export FLOWER_DATA_CACHE=data_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
export PATH="$PATH:/usr/lib/wsl/lib"

GATE=runs/mtp_screen_450m/summary.json
log() { echo "[queue-wfu] $* -- $(date -Is)"; }

log "waiting for the MTP screen: $GATE"
idle=0
while [ ! -f "$GATE" ]; do
  if pgrep -f "flower.sweep" > /dev/null; then
    idle=0
  else
    idle=$((idle + 1))
    if [ "$idle" -ge 2 ]; then log "nothing training for 2 polls; proceeding"; break; fi
  fi
  sleep 300
done
log "gate cleared"

if command -v nvidia-smi > /dev/null; then
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 4096 ] && { log "VRAM at ${used} MiB"; break; }
    sleep 30
  done
fi
sleep 30

if [ -f runs/window_followup_450m/summary.json ]; then
  log "SKIP window follow-up (already complete)"
else
  log "=== window follow-up (3 arms, ~9.7h) ==="
  uv run python -m flower.sweep \
    --config configs/window_followup_450m.yaml \
    --output-dir runs/window_followup_450m
  log "window follow-up exit=$?"
fi

log "FINISHED"
cat <<'NOTE'
[queue-wfu] the decision is w512_4k_isotime vs w2048_4k:
  both run the same wall-clock (4784 x 1.196 ~= 4000 steps of the control).
  If the gap is still near -0.062, the window win is real and the production
  default should change to 512. If it has decayed toward zero, the 1500-step
  screen measured a warmup transient and the default should stay at 2048.
  w256 brackets the optimum, which 2048->1024->512 never turned over.
NOTE
