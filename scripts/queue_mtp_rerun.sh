#!/usr/bin/env bash
# MTP screen RE-RUN, after the eval-loss leak fix.
#
# WHY THIS EXISTS
#   The first MTP screen (runs/mtp_screen_450m, 2026-08-13) is INVALID. Its
#   val_bpb numbers (1.128 -> 2.036 -> 3.069 with 0/1/2 heads) are not model
#   quality: the auxiliary t+2/t+3 losses were being added to the EVAL loss.
#   The fused CE path is gated on `self.training`, so evaluation always fell
#   into the eager branch, which added `mtp_weight * mtp_loss` unconditionally.
#   Fixed in flower/models/base.py; guarded by tests/test_mtp_eval_loss.py.
#
#   New output dir rather than reusing the old one: the invalid run is kept as
#   the evidence for the diagnosis (and its arithmetic is what identified the
#   cause — implied aux losses of 1.61x/1.83x the main loss, monotone in offset).
#
# ORDER
#   Gated on the window follow-up, which is the higher-value question: that one
#   is chasing a measured -0.062 val_bpb at equal wall-clock, the largest effect
#   found in this whole investigation. MTP is a ~5%/head cost still looking for
#   any payoff at all.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

export FLOWER_DATA_CACHE=data_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
export PATH="$PATH:/usr/lib/wsl/lib"

GATE=runs/window_followup_450m/summary.json
log() { echo "[queue-mtp2] $* -- $(date -Is)"; }

log "waiting for the window follow-up: $GATE"
idle=0
while [ ! -f "$GATE" ]; do
  if pgrep -f "flower.sweep" > /dev/null; then idle=0; else
    idle=$((idle + 1))
    [ "$idle" -ge 2 ] && { log "nothing training for 2 polls; proceeding"; break; }
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

# Refuse to run if the fix is not in place — otherwise this silently reproduces
# the invalid screen and wastes ~9h.
if ! uv run python -m pytest tests/test_mtp_eval_loss.py -q > /dev/null 2>&1; then
  log "ABORT: tests/test_mtp_eval_loss.py fails; the eval-loss leak fix is not in place"
  exit 1
fi
log "eval-loss leak fix verified"

if [ -f runs/mtp_screen_450m_fixed/summary.json ]; then
  log "SKIP MTP re-run (already complete)"
else
  log "=== MTP screen re-run (5 arms, ~9.3h) ==="
  uv run python -m flower.sweep \
    --config configs/mtp_screen_450m.yaml \
    --output-dir runs/mtp_screen_450m_fixed
  log "MTP re-run exit=$?"
fi
log "FINISHED"
