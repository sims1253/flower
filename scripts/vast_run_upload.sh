#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/vast_common.sh"
load_vast_defaults

dry_run="false"
instance_id=""
remote_cmd="uv run python -m flower.train --config configs/research_sweep_remote.yaml --device auto"
forward_hf_token="false"
ssh_host_override=""
ssh_port_override=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance-id) instance_id="$2"; shift 2 ;;
    --cmd) remote_cmd="$2"; shift 2 ;;
    --hf-token) forward_hf_token="true"; shift ;;
    --ssh-host) ssh_host_override="$2"; shift 2 ;;
    --ssh-port) ssh_port_override="$2"; shift 2 ;;
    --dry-run) dry_run="true"; shift ;;
    -h|--help) echo "Usage: $0 --instance-id ID [--cmd REMOTE_COMMAND] [--hf-token] [--ssh-host HOST --ssh-port PORT] [--dry-run]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$instance_id" ]] || { echo "ERROR: --instance-id is required" >&2; exit 2; }
if [[ "$dry_run" != "true" ]]; then ensure_vast_api_key; fi
archive="$(mktemp /tmp/flower-vast.XXXXXX.tar.gz)"
trap 'rm -f "$archive"' EXIT
make_repo_archive "$archive"
if [[ "$dry_run" == "true" ]]; then
  echo "DRY RUN: would upload repo archive excluding venv/caches/runs/secrets to instance $instance_id:$VAST_REMOTE_DIR"
  echo "DRY RUN: would run remote setup with uv, Python fallback $VAST_PYTHON_FALLBACK, then: $remote_cmd"
  exit 0
fi
if [[ -n "$ssh_host_override" && -n "$ssh_port_override" ]]; then
  user="root"
  host="$ssh_host_override"
  port="$ssh_port_override"
else
  ssh_url="$(python3 "$REPO_ROOT/scripts/vast_instance_info.py" ssh-url "$instance_id")"
  read -r user host port < <(ssh_args_from_url "$ssh_url")
fi
build_ssh_opts "$port"
ssh "${ssh_opts[@]}" "$user@$host" "mkdir -p '$VAST_REMOTE_DIR'"
scp "${scp_opts[@]}" "$archive" "$user@$host:/tmp/flower.tar.gz"
ssh "${ssh_opts[@]}" "$user@$host" "tar -xzf /tmp/flower.tar.gz -C '$VAST_REMOTE_DIR' && rm /tmp/flower.tar.gz"
setup_cmd="$(remote_setup_command)"
ssh "${ssh_opts[@]}" "$user@$host" "export VAST_PYTHON_FALLBACK='$VAST_PYTHON_FALLBACK'; $setup_cmd"

# If --hf-token was passed and HF_TOKEN is set locally, push it to /root/.hf_env
# (chmod 600) and arrange for the launch heredoc to source it. Token never appears
# in command-line args so it doesn't show up in `ps`.
if [[ "$forward_hf_token" == "true" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: --hf-token requested but HF_TOKEN is not set in the local environment" >&2
    exit 2
  fi
  printf 'HF_TOKEN=%s\n' "$HF_TOKEN" | ssh "${ssh_opts[@]}" "$user@$host" "cat > /root/.hf_env && chmod 600 /root/.hf_env"
  export VAST_HF_TOKEN_INJECT='set -a; source /root/.hf_env; set +a'
fi
remote_cmd="export PATH=\"\$HOME/.local/bin:\$PATH\"; ${VAST_HF_TOKEN_INJECT:-true}; $remote_cmd"
printf -v remote_dir_q '%q' "$VAST_REMOTE_DIR"
printf -v run_dir_q '%q' "$VAST_RUN_DIR"
printf -v remote_cmd_q '%q' "$remote_cmd"
ssh "${ssh_opts[@]}" "$user@$host" "bash -s -- $remote_dir_q $run_dir_q $remote_cmd_q" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
run_dir="$2"
remote_cmd="$3"
cd "$remote_dir"
mkdir -p "$run_dir"
# Ensure uv (installed under $HOME/.local/bin during setup) is on PATH inside the
# nohup'd login shell. `bash -lc` would normally source ~/.profile, but on Vast
# images the default profile does not extend PATH for /root/.local/bin.
# HF_TOKEN setup, when requested, is folded into `remote_cmd` before quoting so
# the token never appears on a command line.
nohup bash -lc "$remote_cmd" > "$run_dir/remote.log" 2>&1 &
printf '%s\n' "$!" > "$run_dir/remote.pid"
REMOTE
