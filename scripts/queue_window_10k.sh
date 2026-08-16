#!/usr/bin/env bash
# Window 512 at production horizon (10k steps) — the decision run.
# See configs/window_10k_450m.yaml for why. Short version: the equal-wall-clock
# quality edge decayed -0.062 (1500 steps) -> -0.024 (4000). Production is 10k.
# The throughput edge (1.196x) does not decay and stands regardless.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export FLOWER_DATA_CACHE=data_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
export PATH="$PATH:/usr/lib/wsl/lib"
log() { echo "[queue-w10k] $* -- $(date -Is)"; }

# Do not start on top of anything else: one job at a time on this card.
#
# Match only the PYTHON processes, not any command line that merely CONTAINS
# the string. A bare `pgrep -f "flower.sweep"` also matches the shell wrapper of
# an interactive status check that greps for that name, so this loop would spin
# forever every time someone looked at it. Verified: the first launch of this
# script hung on exactly that.
gpu_busy() { pgrep -af "flower\\.sweep|bench_arms" | grep -qE "^[0-9]+ +([^ ]*/)?(python[0-9.]*|uv)( |$)"; }
while gpu_busy; do
  log "waiting for the GPU to free"; sleep 120
done
if command -v nvidia-smi > /dev/null; then
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 4096 ] && { log "VRAM at ${used} MiB"; break; }
    sleep 30
  done
fi

if [ -f runs/window_10k_450m/summary.json ]; then
  log "SKIP (already complete)"
else
  log "=== window 10k decision run (2 arms, ~25.6h) ==="
  uv run python -m flower.sweep \
    --config configs/window_10k_450m.yaml \
    --output-dir runs/window_10k_450m
  log "exit=$?"
fi
log "FINISHED"
cat <<'NOTE'
[queue-w10k] reading it:
  w512_10k_isotime (11960 steps) vs w2048_10k (10000) run the same wall-clock.
    gap <= -0.02   -> the edge survived; switch the production default to 512.
    gap ~   0.00   -> quality-neutral; STILL switch, because 512 is ~20% faster
                      at parity and that does not decay.
    gap >   0.00   -> the decay went past zero; 512 is a short-horizon artifact
                      and the default stays at 2048.
  Check both arms' realised tok/s first: if the box was contended, the iso-time
  premise broke and the comparison needs rescaling (this happened on the 4k pair).
NOTE
