#!/usr/bin/env bash
# End-to-end vast.ai sweep launcher.
#
# Replaces the multi-step dance of:
#   1. vastai search offers
#   2. vast_create.sh (with --bid-price)
#   3. wait until SSH is up (and port 22 actually mapped)
#   4. vast_run_upload.sh --hf-token
#   5. tmux launch with PYTORCH env vars
#   6. start tensorboard in another tmux
#   7. fish out the right SSH/TB endpoint URLs
#
# Usage:
#   bash scripts/vast_launch_sweep.sh \
#     --offer-id 33666510 \
#     --bid-price 0.30 \
#     --sweep-config configs/sweep2_phase0_tokenizer.yaml \
#     --output-dir runs/vast/sweep2_phase0 \
#     --steps 10000
#
# Requires: VAST_API_KEY in env, HF_TOKEN in env, SSH key at ~/.ssh/id_ed25519.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vast_common.sh"
load_vast_defaults

offer_id=""
bid_price="0.30"
max_price="0.50"
sweep_config="configs/sweep2_phase0_tokenizer.yaml"
output_dir="runs/vast/sweep2_phase0"
steps="10000"
gpus="0,1"
forward_hf_token="true"
start_tb="true"
tb_port="6006"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --offer-id) offer_id="$2"; shift 2 ;;
    --bid-price) bid_price="$2"; shift 2 ;;
    --max-price) max_price="$2"; shift 2 ;;
    --sweep-config) sweep_config="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --no-hf-token) forward_hf_token="false"; shift ;;
    --no-tb) start_tb="false"; shift ;;
    --tb-port) tb_port="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$offer_id" ]] || { echo "ERROR: --offer-id required" >&2; exit 2; }
ensure_vast_api_key
[[ "$forward_hf_token" != "true" || -n "${HF_TOKEN:-}" ]] || { echo "ERROR: HF_TOKEN must be set in env (or pass --no-hf-token)" >&2; exit 2; }

log() { printf '\033[1;36m[launch]\033[0m %s\n' "$*"; }

# ---------- Step 1: create instance with bid ----------
log "creating instance from offer $offer_id at bid \$$bid_price/h (max \$$max_price)"
create_out=$(bash "$SCRIPT_DIR/vast_create.sh" \
  --offer-id "$offer_id" \
  --max-price "$max_price" \
  --offer-type bid \
  --bid-price "$bid_price" \
  --yes 2>&1)
instance_id=$(echo "$create_out" | python3 -c "import json,sys,re
for l in sys.stdin:
  l=l.strip()
  if not l.startswith('{'): continue
  try:
    d=json.loads(l)
    if 'new_contract' in d: print(d['new_contract']); break
  except Exception: pass
")
[[ -n "$instance_id" ]] || { echo "ERROR: failed to extract instance_id from create output:" >&2; echo "$create_out" >&2; exit 1; }
log "created instance_id=$instance_id"

# ---------- Step 2: wait for SSH endpoint to be reachable ----------
log "waiting for instance to come up (loading image, mapping ports)..."
ssh_host=""; ssh_port=""
for i in $(seq 1 60); do
  status_json=$(uvx --from vastai python "$SCRIPT_DIR/vast_instance_info.py" status "$instance_id" 2>/dev/null)
  parsed=$(echo "$status_json" | python3 -c "
import json, sys, re
data = sys.stdin.read()
data = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', data)
try: d = json.loads(data)
except Exception: print('parse_fail'); sys.exit()
actual = d.get('actual_status','?')
ports = d.get('ports') or {}
host = d.get('public_ipaddr','')
host_port_22 = ''
if ports.get('22/tcp'):
  host_port_22 = ports['22/tcp'][0].get('HostPort','')
print(f'{actual}|{host}|{host_port_22}')
" 2>/dev/null)
  IFS='|' read -r actual host port22 <<<"$parsed"
  if [[ "$actual" == "running" ]]; then
    # Try direct SSH first if port 22 is exposed.
    if [[ -n "$port22" && -n "$host" ]]; then
      if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p "$port22" "root@$host" 'echo READY' 2>/dev/null | grep -q READY; then
        ssh_host="$host"; ssh_port="$port22"
        log "SSH ready (direct): $ssh_host:$ssh_port"
        break
      fi
    fi
    # Always also try vast jumphost — many vast images don't expose direct port 22.
    ssh_url=$(uvx --from vastai python "$SCRIPT_DIR/vast_instance_info.py" ssh-url "$instance_id" 2>/dev/null | tr -d '\r\n')
    if [[ -n "$ssh_url" ]]; then
      jh_user=$(echo "$ssh_url" | sed -E 's|ssh://([^@]+)@.*|\1|')
      jh_host=$(echo "$ssh_url" | sed -E 's|ssh://[^@]+@([^:]+):.*|\1|')
      jh_port=$(echo "$ssh_url" | sed -E 's|.*:([0-9]+)$|\1|')
      if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p "$jh_port" "$jh_user@$jh_host" 'echo READY' 2>/dev/null | grep -q READY; then
        ssh_host="$jh_host"; ssh_port="$jh_port"
        log "SSH ready (jumphost): $ssh_host:$ssh_port"
        break
      fi
    fi
  fi
  printf '.'
  sleep 15
done
echo
[[ -n "$ssh_host" ]] || { echo "ERROR: SSH never came up after 15 minutes" >&2; exit 1; }

ssh_opts=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p "$ssh_port")
ssh_remote() { ssh "${ssh_opts[@]}" "root@$ssh_host" "$@"; }

# ---------- Step 3: upload code ----------
log "uploading repo to $ssh_host:/workspace/flower"
archive=$(mktemp /tmp/flower-vast.XXXXXX.tar.gz)
trap 'rm -f "$archive"' EXIT
make_repo_archive "$archive"
ssh_remote "mkdir -p /workspace/flower"
scp -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -P "$ssh_port" "$archive" "root@$ssh_host:/tmp/flower.tar.gz" >/dev/null
ssh_remote "tar -xzf /tmp/flower.tar.gz -C /workspace/flower && rm /tmp/flower.tar.gz"

# ---------- Step 4: HF_TOKEN to /root/.hf_env (chmod 600, never logged) ----------
if [[ "$forward_hf_token" == "true" ]]; then
  log "pushing HF_TOKEN to remote (encrypted in transit, chmod 600 on disk)"
  printf 'HF_TOKEN=%s\n' "$HF_TOKEN" | ssh_remote "cat > /root/.hf_env && chmod 600 /root/.hf_env"
fi

# ---------- Step 5: ensure uv installed + project synced ----------
log "running setup (uv install + project sync)"
setup_cmd=$(remote_setup_command)
ssh_remote "export VAST_PYTHON_FALLBACK='$VAST_PYTHON_FALLBACK'; $setup_cmd" 2>&1 | tail -3

# ---------- Step 6: launch sweep in detached tmux ----------
hf_inject="true"
[[ "$forward_hf_token" == "true" ]] && hf_inject="set -a; source /root/.hf_env; set +a"

launch_cmd="export PATH=\$HOME/.local/bin:\$PATH; \
${hf_inject}; \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
cd /workspace/flower; \
uv run python -m flower.sweep_parallel \
  --config $sweep_config \
  --gpus $gpus \
  --output-dir $output_dir \
  --steps $steps \
  --device cuda 2>&1 | tee runs/vast/remote.log"

log "launching sweep in tmux session 'sweep'"
ssh_remote "tmux kill-session -t sweep 2>/dev/null; mkdir -p /workspace/flower/runs/vast; tmux new-session -d -s sweep '$launch_cmd'"

# ---------- Step 7: optional tensorboard in second tmux ----------
if [[ "$start_tb" == "true" ]]; then
  tb_cmd="export PATH=\$HOME/.local/bin:\$PATH; cd /workspace/flower; uv run tensorboard --logdir $output_dir --host 0.0.0.0 --port $tb_port"
  ssh_remote "tmux kill-session -t tb 2>/dev/null; tmux new-session -d -s tb '$tb_cmd'"
  log "tensorboard launched in tmux session 'tb' on remote port $tb_port"
fi

# ---------- Step 8: print useful URLs ----------
cat <<EOF

\033[1;32m[launch] DONE\033[0m
  instance_id=$instance_id
  SSH: ssh -p $ssh_port root@$ssh_host

  Live logs:
    ssh -p $ssh_port root@$ssh_host 'tail -f /workspace/flower/runs/vast/remote.log'
    ssh -p $ssh_port root@$ssh_host 'tmux attach -t sweep'

  TensorBoard (if started; port not auto-mapped on this image, use SSH tunnel):
    ssh -L $tb_port:localhost:$tb_port -p $ssh_port root@$ssh_host
    then open http://localhost:$tb_port/

  Pull metrics back:
    bash scripts/vast_pull.sh --instance-id $instance_id --remote-path $output_dir --local-path runs/local_pull/

  Destroy instance when done:
    bash scripts/vast_stop_destroy.sh --instance-id $instance_id --action destroy --yes
EOF
