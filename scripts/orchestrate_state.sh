#!/usr/bin/env bash
# Idempotent state probe for the autonomous Sweep 2 orchestration.
#
# Prints a single-line state token that the autonomous loop uses to decide
# what action to take next. States are:
#   PHASE1_RUNNING   — Phase 1 sweep alive, not yet done
#   PHASE1_DONE      — Phase 1 finished (summary.json present), no Phase 2 launched yet
#   PHASE2_RUNNING   — Phase 2 launched, not yet done
#   PHASE2_DONE      — Phase 2 finished
#   ALL_PULLED       — final pull to local complete (sentinel file present)
#   INSTANCE_DEAD    — instance stopped/exited (need to restart or accept loss)
#   UNKNOWN          — couldn't reach instance
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE_ID="${INSTANCE_ID:-36458346}"
SSH_HOST="${SSH_HOST:-ssh9.vast.ai}"
SSH_PORT="${SSH_PORT:-18346}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH="ssh -o ConnectTimeout=10 -i $SSH_KEY -o IdentitiesOnly=yes -p $SSH_PORT root@$SSH_HOST"

instance_status=$(uvx --from vastai python "$REPO_ROOT/scripts/vast_instance_info.py" status $INSTANCE_ID 2>/dev/null \
  | python3 -c "
import json, sys, re
data = re.sub(r'[\\x00-\\x1f\\x7f-\\x9f]', ' ', sys.stdin.read())
try: print(json.loads(data).get('actual_status','?'))
except Exception: print('parse_fail')
" 2>/dev/null)

if [[ "$instance_status" != "running" ]]; then
  echo "INSTANCE_DEAD ($instance_status)"
  exit 0
fi

if [[ -f "$REPO_ROOT/runs/local_pull/SWEEP2_COMPLETE" ]]; then
  echo "ALL_PULLED"
  exit 0
fi

# Probe remote
remote_state=$($SSH "
p1_summary=\$(ls /workspace/flower/runs/vast/sweep2_phase1/summary.json 2>/dev/null)
p1_count=\$(ls /workspace/flower/runs/vast/sweep2_phase1/*.metrics.json 2>/dev/null | wc -l)
p2_summary=\$(ls /workspace/flower/runs/vast/sweep2_phase2/summary.json 2>/dev/null)
p2_count=\$(ls /workspace/flower/runs/vast/sweep2_phase2/*.metrics.json 2>/dev/null | wc -l)
sweep_alive=\$(tmux has-session -t sweep 2>/dev/null && echo Y || echo N)
echo \"p1_summary=\$p1_summary p1_count=\$p1_count p2_summary=\$p2_summary p2_count=\$p2_count sweep_alive=\$sweep_alive\"
" 2>/dev/null)

if [[ -z "$remote_state" ]]; then
  echo "UNKNOWN (ssh probe failed)"
  exit 0
fi

eval "$remote_state"

if [[ -n "$p2_summary" ]]; then
  echo "PHASE2_DONE (p2_count=$p2_count)"
elif [[ -n "$p1_summary" ]]; then
  # Phase 1 done. Either Phase 2 not launched, or Phase 2 in progress.
  if [[ "$sweep_alive" == "Y" && "$p2_count" -gt 0 ]]; then
    echo "PHASE2_RUNNING (p2_count=$p2_count)"
  elif [[ "$sweep_alive" == "Y" ]]; then
    echo "PHASE2_RUNNING (warming up)"
  else
    echo "PHASE1_DONE (p1_count=$p1_count, ready for Phase 2 launch)"
  fi
else
  if [[ "$sweep_alive" == "Y" ]]; then
    echo "PHASE1_RUNNING (p1_count=$p1_count/20)"
  else
    echo "PHASE1_DEAD (p1_count=$p1_count, sweep tmux gone)"
  fi
fi
