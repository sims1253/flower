#!/usr/bin/env bash
# Window + MTP quality screens, chained behind the CM/Aurora screen.
#
# WHY THIS IS A SEPARATE SCRIPT AND NOT MORE STAGES IN queue_investigations.sh
#   It is also stages 5 and 6 of that script — but the copy of it that is
#   CURRENTLY RUNNING (armed 2026-08-12T09:21) will never execute them. Bash
#   holds an open fd on the script's inode and keeps reading that inode; editing
#   the file with a tool that writes-and-renames leaves the running shell on the
#   old, now-deleted inode. Verified directly:
#       /proc/<pid>/fd/255 -> scripts/queue_investigations.sh (deleted)
#   So the live queue will finish stage 4 and exit. This script picks up where
#   it stops, without touching it — restarting the live queue mid-CM-screen to
#   pick up an edit would throw away hours of a 13.2h run for no reason.
#
#   Both paths are guarded by the same `summary.json` existence checks, so if
#   queue_investigations.sh is ever re-armed from scratch (after a reboot, say)
#   it will skip whatever this script already finished. Running both is safe;
#   running neither is not.
#
# WHAT IT WAITS FOR
#   runs/comp_muon_screen_450m/summary.json — written by flower.sweep only after
#   the last CM arm completes. ~13.2h at the measured 1.89h/arm x 7 arms.
#
# ORDER: WINDOW BEFORE MTP
#   The window screen answers the bigger question. Stage 3 measured window 512
#   at 1.268x throughput (cross-validated at 1.256x by the warmup bench, both
#   ~1% spread) — the largest lever found in the whole unused-feature sweep, and
#   one that applies to every future run including the MTP screen itself. MTP is
#   a ~5%/head cost looking for a payoff; the window is a 27% win looking for a
#   price. Answer the second one first.
#
# USAGE
#   nohup bash scripts/queue_screens.sh > runs_queue_screens.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

export FLOWER_DATA_CACHE=data_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
export PATH="$PATH:/usr/lib/wsl/lib"

GATE=runs/comp_muon_screen_450m/summary.json

log() { echo "[queue-screens] $* -- $(date -Is)"; }

log "waiting for the CM/Aurora screen: $GATE"
idle=0
while [ ! -f "$GATE" ]; do
  # Same fallback as the other waiters: if nothing is training for two
  # consecutive polls the chain died rather than finished, and an abandoned
  # card is still ours. Poll at 5 min — this is a 13h wait, not a 5 min one.
  if pgrep -f "flower.sweep|bench_arms" > /dev/null; then
    idle=0
  else
    idle=$((idle + 1))
    if [ "$idle" -ge 2 ]; then log "nothing training for 2 polls; proceeding"; break; fi
  fi
  sleep 300
done
log "gate cleared"

if command -v nvidia-smi > /dev/null; then
  log "waiting for VRAM to drop below 4 GB"
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 4096 ] && { log "VRAM at ${used} MiB"; break; }
    sleep 30
  done
fi
sleep 30

# --- window: quality price of the 1.268x -------------------------------------
if [ -f runs/window_screen_450m/summary.json ]; then
  log "SKIP window screen (already complete)"
else
  log "=== window screen (4 arms, ~8.1h) ==="
  uv run python -m flower.sweep \
    --config configs/window_screen_450m.yaml \
    --output-dir runs/window_screen_450m
  log "window screen exit=$?"
fi

sleep 30

# --- MTP: payoff for the measured ~5%/head -----------------------------------
if [ -f runs/mtp_screen_450m/summary.json ]; then
  log "SKIP MTP screen (already complete)"
else
  log "=== MTP screen (5 arms, ~9.3h) ==="
  uv run python -m flower.sweep \
    --config configs/mtp_screen_450m.yaml \
    --output-dir runs/mtp_screen_450m
  log "MTP screen exit=$?"
fi

log "ALL SCREENS FINISHED"
cat <<'NOTE'
[queue-screens] how to read these:

WINDOW (runs/window_screen_450m)
  The decision arm is w512_isotime: 1902 steps = the SAME wall-clock as
  w2048_control at 1500. If it beats the control's val_bpb, the shorter window
  is strictly better under a compute budget and the production default should
  change. w512/w1024 at matched steps give the pure per-step quality cost.
  Check each arm's tok/s against the bench (75,904 / 69,472 / 59,854) before
  trusting the iso-time sizing.
  This is vanilla_local only — do NOT transfer it to a memory variant, whose
  whole thesis is that it uses context differently.

MTP (runs/mtp_screen_450m)
  Eval uses the main t+1 head alone, so val_bpb is directly comparable with no
  adjustment. mtp2_isotime (1360 steps) is the deployment question; mtp1/mtp2 at
  1500 are MTP's actual sample-efficiency claim. mtp2_w025 exists to tell
  "MTP does not help here" apart from "mtp_weight 0.5 was too strong", since
  that default was never tuned against anything.

BOTH
  Treat gaps under ~0.01 val_bpb as inconclusive: the 600-step reseed band was
  0.01282 and the 1500-step band has never been measured. Compare loss-vs-step
  curves (log_interval 25), not just the final number.
NOTE
