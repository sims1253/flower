#!/usr/bin/env bash
# Queue the CM / Aurora / unused-feature investigations behind the TST rerun.
#
# WHAT IT WAITS FOR
#   runs/tst_r50_rerun/summary.json — the LAST item of the existing queue chain
#   (scripts/queue_tst_r50.sh, itself gated on the per-head confirmation). When
#   that file appears the whole prior chain is done and the card is ours.
#   Polling a file rather than a PID is the house pattern: it survives this
#   script being restarted, and it does not race with a job that has finished
#   computing but not yet flushed.
#
#   Secondary gate: even after the file appears, wait for VRAM to actually drop.
#   summary.json is written before the process exits and frees its allocations,
#   and starting a 450M compile against a card that still holds 25 GB is how you
#   get a WSL2 host-RAM spill instead of a clean OOM.
#
# ORDERING: CHEAPEST DIAGNOSTIC FIRST
#   Not arbitrary. bench_optimizer_step is ~5 min and needs no torch.compile,
#   and it prices the two optimizers the 4-hour screen is built around. On CPU
#   Aurora measured 2.4x the control's optimizer step; if that holds on GPU it
#   is ~4% of step time, which is worth knowing BEFORE committing two Aurora
#   arms to an overnight screen rather than discovering it inside one.
#   Then the throughput benches (~1 h), then the quality screen (~4 h) last so
#   the long job runs unattended.
#
# ONE JOB AT A TIME
#   Established the hard way: benchmarking alongside a training run cost ~3%
#   throughput (58.9k -> 57.1k tok/s), and heavier overlap risks pushing peak
#   VRAM past 32 GB into WSL2's silent host-RAM spill, where throughput
#   collapses by an order of magnitude with NO error. Every stage here is
#   sequential and separated by a VRAM-drain wait.
#
# IDEMPOTENT
#   Each stage skips if its output already exists, so this can be re-armed after
#   a reboot or a kill without redoing finished work.
#
# USAGE
#   nohup bash scripts/queue_investigations.sh > runs_queue_investigations.log 2>&1 &
set -uo pipefail
# `|| exit` rather than the bare `cd` the older queue scripts use: every path
# below is repo-relative, so a failed cd would run the whole queue somewhere else.
cd "$(dirname "$0")/.." || exit 1

export FLOWER_DATA_CACHE=data_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
# After a reboot nvidia-smi is not on PATH in a non-login shell — it lives in
# the WSL driver dir. Without this the VRAM-drain check below reads an empty
# string, falls back to its 99999 default and burns its full 30-minute budget
# before proceeding. Observed on the 2026-08-12 reboot.
export PATH="$PATH:/usr/lib/wsl/lib"

GATE=runs/tst_r50_rerun/summary.json
BENCH_CFG=configs/perf_bench_450m.yaml
SCREEN_CFG=configs/comp_muon_screen_450m.yaml
OUT=runs/comp_muon_screen_450m
LOGDIR=docs/profiling
mkdir -p "$LOGDIR"

log() { echo "[queue-inv] $* -- $(date -Is)"; }

# --- wait for the prior chain -------------------------------------------------
log "waiting for the TST rerun to finish: $GATE"
idle=0
while [ ! -f "$GATE" ]; do
  # Fallback: if no sweep is alive for two consecutive polls, the chain died
  # rather than finished. Proceed anyway — an abandoned card is still ours, and
  # blocking forever on a file that will never appear is the worse failure.
  if pgrep -f "flower.sweep" > /dev/null; then
    idle=0
  else
    idle=$((idle + 1))
    if [ "$idle" -ge 2 ]; then log "no sweep alive for 2 polls; proceeding"; break; fi
  fi
  sleep 120
done
log "gate cleared"

# --- wait for VRAM to actually drain -----------------------------------------
# summary.json lands before the process exits and frees its allocations.
if ! command -v nvidia-smi > /dev/null; then
  log "WARNING: nvidia-smi not found; skipping the VRAM-drain wait"
  sleep 60
else
  log "waiting for VRAM to drop below 4 GB"
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 4096 ] && { log "VRAM at ${used} MiB"; break; }
    sleep 30
  done
  sleep 30
fi

# --- stage 1: optimizer step cost (fast, no compile) -------------------------
S1=$LOGDIR/bench_optimizer_step.txt
if [ -f "$S1" ]; then
  log "SKIP stage 1 (exists: $S1)"
else
  log "=== stage 1/6: optimizer step cost (CM / Aurora / per-head) ==="
  # --step-ms 1206 is the measured 450M full-step wall-clock at accum 4 from
  # docs/profiling/speedup_results.md, so the '% of step' column is meaningful.
  uv run python scripts/bench_optimizer_step.py \
    --config "$BENCH_CFG" --step-ms 1206 --repeats 5 \
    --arms perf_control,opt_per_head,opt_cm_isotropic,opt_cm_full,opt_aurora \
    2>&1 | tee "$S1"
  log "stage 1 exit=${PIPESTATUS[0]}"
fi
sleep 30

# --- stage 2: attention-window warmup recompile safety -----------------------
# --plan-only needs no GPU and already flags quantize 250 as unsafe (9 distinct
# windows vs cache_size_limit 8), so the real run uses 400 (6 windows).
S2=$LOGDIR/bench_attn_warmup.txt
if [ -f "$S2" ]; then
  log "SKIP stage 2 (exists: $S2)"
else
  log "=== stage 2/6: attention window warmup (recompile safety + throughput) ==="
  uv run python scripts/bench_attn_warmup.py \
    --config "$BENCH_CFG" --arm perf_control \
    --start 512 --warmup-steps 2000 --quantize 400 --accum 4 \
    2>&1 | tee "$S2"
  log "stage 2 exit=${PIPESTATUS[0]}"
fi
sleep 30

# --- stage 3: throughput + VRAM of the unused features -----------------------
S3=$LOGDIR/bench_perf_arms.txt
if [ -f "$S3" ]; then
  log "SKIP stage 3 (exists: $S3)"
else
  log "=== stage 3/6: MTP / fp8_lm_head / window cost curve ==="
  # accum 4 keeps each arm ~2 min of steady state; relative gaps hold and the
  # wall-clock shrinks. repeats 3 because run-to-run drift here is ~2% and a
  # single timing is not evidence for a single-digit claim.
  uv run python scripts/bench_arms.py \
    --config "$BENCH_CFG" --repeats 3 --accum 4 \
    --arms perf_control,mtp_1_head,mtp_2_heads,mtp_2_heads_unfused,fp8_lm_head,window_512,window_1024,window_4096 \
    2>&1 | tee "$S3"
  log "stage 3 exit=${PIPESTATUS[0]}"
fi
sleep 30

# --- stage 3b: re-measure the arms whose spread made them unusable -----------
# The first stage-3 pass returned two arms with spread ABOVE 100%:
#   mtp_2_heads_unfused  33,630 tok/s  120.1%  (min 14,616 / max 55,004)
#   window_4096          39,057 tok/s  125.2%  (min  3,654 / max 52,567)
# A 10x low outlier in one repeat is an autotune/allocator stall that --warmup 3
# did not absorb, not a property of the arm — every other arm on the same pass
# came in under 5.1%. The reported median is therefore meaningless for these two
# (window_4096's true value is probably near its 52,567 max). bench_arms itself
# says a gap smaller than the spread is not evidence; here the spread swamps the
# gap entirely. Re-run just those two with more warmup and more repeats. The
# control is included so the comparison is inside one invocation, as required.
S3B=$LOGDIR/bench_perf_arms_rerun.txt
if [ -f "$S3B" ]; then
  log "SKIP stage 3b (exists: $S3B)"
else
  log "=== stage 3b/6: re-measure high-spread arms (mtp_2_heads_unfused, window_4096) ==="
  uv run python scripts/bench_arms.py \
    --config "$BENCH_CFG" --repeats 5 --warmup 6 --accum 4 \
    --arms perf_control,mtp_2_heads_unfused,window_4096 \
    2>&1 | tee "$S3B"
  log "stage 3b exit=${PIPESTATUS[0]}"
fi
sleep 30

# --- stage 4: the quality screen (long) --------------------------------------
if [ -f "$OUT/summary.json" ]; then
  log "SKIP stage 4 (exists: $OUT/summary.json)"
else
  log "=== stage 4/6: Compositional Muon + Aurora screen (7 arms x 1500 steps) ==="
  uv run python -m flower.sweep --config "$SCREEN_CFG" --output-dir "$OUT"
  log "stage 4 exit=$?"
fi

sleep 30

# --- stage 5: attention window quality/throughput tradeoff -------------------
# Queued because stage 3 found the largest throughput lever of the sweep:
# window 512 is 1.268x, cross-validated at 1.256x by the warmup bench, both at
# ~1% spread. The throughput side is settled; the quality side is unmeasured.
# The screen's decision arm (w512_isotime, 1902 steps) runs the SAME wall-clock
# as the control, so beating it means the shorter window is strictly better
# under a compute budget.
if [ -f runs/window_screen_450m/summary.json ]; then
  log "SKIP stage 5 (exists: runs/window_screen_450m/summary.json)"
else
  log "=== stage 5/6: attention window screen (4 arms, ~8.1h) ==="
  uv run python -m flower.sweep \
    --config configs/window_screen_450m.yaml \
    --output-dir runs/window_screen_450m
  log "stage 5 exit=$?"
fi

sleep 30

# --- stage 6: MTP quality ----------------------------------------------------
# ~5% throughput and ~0.46 GB per head, measured. Eval uses the main t+1 head
# only, so val_bpb is directly comparable across arms with no adjustment.
if [ -f runs/mtp_screen_450m/summary.json ]; then
  log "SKIP stage 6 (exists: runs/mtp_screen_450m/summary.json)"
else
  log "=== stage 6/6: MTP screen (5 arms, ~9.3h) ==="
  uv run python -m flower.sweep \
    --config configs/mtp_screen_450m.yaml \
    --output-dir runs/mtp_screen_450m
  log "stage 6 exit=$?"
fi

log "ALL QUEUED WORK FINISHED"
cat <<'NOTE'
[queue-inv] analyse the screen with:
  PYTHONPATH=. uv run python scripts/analyze_speedup_screen.py \
      --run-dir runs/comp_muon_screen_450m --control cm_baseline \
      --seed-dir runs/speedup_screen_450m_seed1

[queue-inv] reading the screen (from the config header):
  - The metric is STEPS-TO-TARGET-LOSS, not final val_bpb. A 5-10% sample
    efficiency gain sits below the seed band at 1500 steps; compare the
    loss-vs-step curves (log_interval 25).
  - cm_baseline should reproduce muon_screen_450m's muon_baseline (1.12785).
    If it does not, the box drifted and the cross-screen comparison is void.
  - CM has to beat cm_per_head_ref (1.10071), not just cm_baseline. Beating
    only the baseline means the partner whitening bought nothing over the
    scope change that per-head already gets.
NOTE
