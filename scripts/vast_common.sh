#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAST_CONFIG_DEFAULT="$REPO_ROOT/configs/vast_defaults.env"

load_vast_defaults() {
  if [[ -f "$VAST_CONFIG_DEFAULT" ]]; then
    # shellcheck disable=SC1090
    source "$VAST_CONFIG_DEFAULT"
  fi
  : "${VAST_MAX_PRICE:=0.20}"
  : "${VAST_IMAGE:=pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel}"
  : "${VAST_DISK_GB:=40}"
  : "${VAST_QUERY:=verified=true rentable=true rented=false gpu_ram>=16 num_gpus=1 inet_down>=100 inet_up>=20 dph<=${VAST_MAX_PRICE}}"
  : "${VAST_REMOTE_DIR:=/workspace/flower}"
  : "${VAST_RUN_DIR:=runs/vast}"
  : "${VAST_PYTHON_FALLBACK:=3.12}"
  : "${VAST_SSH_KEY:=$HOME/.ssh/id_ed25519}"
}

build_ssh_opts() {
  local port="$1"
  ssh_opts=(-o StrictHostKeyChecking=accept-new -p "$port")
  scp_opts=(-o StrictHostKeyChecking=accept-new -P "$port")
  if [[ -n "${VAST_SSH_KEY:-}" && -f "$VAST_SSH_KEY" ]]; then
    ssh_opts=(-i "$VAST_SSH_KEY" -o IdentitiesOnly=yes "${ssh_opts[@]}")
    scp_opts=(-i "$VAST_SSH_KEY" -o IdentitiesOnly=yes "${scp_opts[@]}")
  fi
}

usage_common() {
  cat <<'USAGE'
Common environment:
  VAST_API_KEY       Required for Vast CLI/API operations. Never written to disk by these scripts.
  VAST_MAX_PRICE     Hourly price ceiling; defaults to 0.20 and cannot exceed --max-price for create/sweep.
  VAST_IMAGE         Docker image for created instances.
  VAST_QUERY         Vast search query. Defaults to verified rentable 1-GPU offers under VAST_MAX_PRICE.
USAGE
}

ensure_vast_api_key() {
  if [[ -z "${VAST_API_KEY:-}" ]]; then
    echo "ERROR: VAST_API_KEY must be set in the environment; it is not stored by this repo." >&2
    exit 2
  fi
}

ensure_vast_cli() {
  if command -v vastai >/dev/null 2>&1; then
    VAST_BIN="vastai"
    return
  fi
  if command -v vast >/dev/null 2>&1; then
    VAST_BIN="vast"
    return
  fi
  if command -v uvx >/dev/null 2>&1; then
    VAST_BIN="uvx --from vastai vastai"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    VAST_BIN="python3 -m vastai"
    if ! python3 -c 'import vastai' >/dev/null 2>&1; then
      echo "ERROR: Vast CLI is not installed. Install it with: python3 -m pip install --user vastai" >&2
      exit 2
    fi
    return
  fi
  echo "ERROR: Vast CLI not found. Install uv/uvx or the vastai Python package." >&2
  exit 2
}

vast() {
  # Intentionally pass the API key only as an argument/environment at runtime; never persist it.
  # shellcheck disable=SC2086
  $VAST_BIN --api-key "$VAST_API_KEY" "$@"
}

require_yes() {
  local yes="false"
  for arg in "$@"; do
    if [[ "$arg" == "--yes" ]]; then
      yes="true"
    fi
  done
  if [[ "$yes" != "true" ]]; then
    echo "ERROR: This action can spend money or destroy resources. Re-run with --yes to confirm." >&2
    exit 2
  fi
}

validate_price() {
  local price="$1"
  python3 - "$price" <<'PY'
import sys
price=float(sys.argv[1])
if price > 0.20:
    raise SystemExit(f"ERROR: max price {price:.4f} exceeds hard safety default 0.20/hr")
PY
}

print_or_run() {
  local dry_run="$1"
  shift
  if [[ "$dry_run" == "true" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

make_repo_archive() {
  local archive="$1"
  (cd "$REPO_ROOT" && tar \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='runs' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    --exclude='.env' \
    --exclude='*.key' \
    --exclude='id_rsa*' \
    -czf "$archive" .)
}

ssh_args_from_url() {
  local ssh_url="$1"
  python3 - "$ssh_url" <<'PY'
import re
import shlex
import sys
from urllib.parse import urlparse

s = sys.argv[1].strip()

user = None
host = None
port = None

# Vast may return a URL-like value, e.g. ssh://root@ssh5.vast.ai:19420.
if s.startswith("ssh://"):
    parsed = urlparse(s)
    user = parsed.username
    host = parsed.hostname
    port = parsed.port
else:
    tokens = shlex.split(s)
    if tokens and tokens[0] == "ssh":
        tokens = tokens[1:]

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "-p" and idx + 1 < len(tokens):
            port = tokens[idx + 1]
            idx += 2
            continue
        if token.startswith("-p") and len(token) > 2:
            port = token[2:]
            idx += 1
            continue
        if "@" in token and not token.startswith("-"):
            user_host = token
            if user_host.startswith("ssh://"):
                parsed = urlparse(user_host)
                user = parsed.username
                host = parsed.hostname
                port = parsed.port or port
            else:
                user, rest = user_host.rsplit("@", 1)
                if rest.startswith("[") and "]" in rest:
                    host_part, _, port_part = rest[1:].partition("]")
                    host = host_part
                    if port_part.startswith(":"):
                        port = port_part[1:]
                elif ":" in rest and rest.count(":") == 1:
                    host, port_part = rest.rsplit(":", 1)
                    if port_part:
                        port = port_part
                else:
                    host = rest
        idx += 1

if not (user and host):
    raise SystemExit(f"Could not parse ssh url: {s}")
print(user, host, str(port or "22"))
PY
}

remote_setup_command() {
  cat <<'REMOTE'
set -euo pipefail
cd /workspace/flower
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [[ -f .python-version ]] && ! uv python find "$(cat .python-version)" >/dev/null 2>&1; then
  echo "Requested Python $(cat .python-version) is unavailable; falling back to ${VAST_PYTHON_FALLBACK:-3.12}."
  uv python install "${VAST_PYTHON_FALLBACK:-3.12}" || true
  UV_PYTHON="${VAST_PYTHON_FALLBACK:-3.12}"
  export UV_PYTHON
fi
uv sync --extra dev
REMOTE
}
