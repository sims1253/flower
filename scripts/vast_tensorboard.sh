#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/vast_common.sh"
load_vast_defaults

instance_id=""
logdir="runs/vast/research_sweep_remote"
port="6006"
bind="127.0.0.1"
dry_run="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance-id) instance_id="$2"; shift 2 ;;
    --logdir) logdir="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --bind) bind="$2"; shift 2 ;;
    --dry-run) dry_run="true"; shift ;;
    -h|--help) echo "Usage: $0 --instance-id ID [--logdir DIR] [--port PORT] [--bind HOST] [--dry-run]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$instance_id" ]] || { echo "ERROR: --instance-id is required" >&2; exit 2; }
[[ "$port" =~ ^[0-9]+$ ]] || { echo "ERROR: --port must be numeric" >&2; exit 2; }

if [[ "$dry_run" != "true" ]]; then ensure_vast_api_key; fi

remote_logdir="$VAST_REMOTE_DIR/$logdir"
remote_log="$VAST_REMOTE_DIR/$VAST_RUN_DIR/tensorboard.log"
remote_pid="$VAST_REMOTE_DIR/$VAST_RUN_DIR/tensorboard.pid"
remote_cmd="export PATH=\"\$HOME/.local/bin:\$PATH\"; cd '$VAST_REMOTE_DIR' && mkdir -p '$VAST_RUN_DIR' && if [ -f '$remote_pid' ] && ps -p \$(cat '$remote_pid') >/dev/null 2>&1; then kill \$(cat '$remote_pid') || true; sleep 1; fi; nohup uv run tensorboard --logdir '$remote_logdir' --host '$bind' --port '$port' > '$remote_log' 2>&1 & printf '%s\n' \$! > '$remote_pid'"

if [[ "$dry_run" == "true" ]]; then
  echo "DRY RUN: would start TensorBoard on instance $instance_id for $remote_logdir at $bind:$port"
  exit 0
fi

ssh_url="$(python3 "$REPO_ROOT/scripts/vast_instance_info.py" ssh-url "$instance_id")"
read -r user host ssh_port < <(ssh_args_from_url "$ssh_url")
build_ssh_opts "$ssh_port"
ssh "${ssh_opts[@]}" "$user@$host" "$remote_cmd"
ssh "${ssh_opts[@]}" "$user@$host" "sleep 2; cat '$remote_pid'; ps -p \$(cat '$remote_pid') -o pid,ppid,stat,cmd; (ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null || true) | grep ':$port ' || true; tail -n 20 '$remote_log' || true"
echo "Tunnel: ssh -L $port:localhost:$port -p $ssh_port $user@$host"
