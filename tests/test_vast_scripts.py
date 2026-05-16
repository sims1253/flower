from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

spec = importlib.util.spec_from_file_location("vast_parse_offers", SCRIPTS / "vast_parse_offers.py")
assert spec is not None and spec.loader is not None
vast_parse_offers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vast_parse_offers)


def test_vast_scripts_require_confirmation_for_spend_or_destroy() -> None:
    for name in ["vast_create.sh", "vast_sweep.sh", "vast_stop_destroy.sh"]:
        text = (SCRIPTS / name).read_text()
        assert "require_yes" in text
        assert "--yes" in text


def test_vast_defaults_keep_price_cap_and_no_secret() -> None:
    text = (ROOT / "configs" / "vast_defaults.env").read_text()
    assert "VAST_MAX_PRICE=0.20" in text
    assert "VAST_API_KEY" not in text


def test_archive_excludes_common_secret_and_large_paths() -> None:
    text = (SCRIPTS / "vast_common.sh").read_text()
    for pattern in [".venv", "__pycache__", ".pytest_cache", "runs", ".env", "*.key", "id_rsa*"]:
        assert f"--exclude='{pattern}'" in text


def test_remote_setup_has_uv_and_python_fallback() -> None:
    text = (SCRIPTS / "vast_common.sh").read_text()
    assert "uv sync --extra dev" in text
    assert "VAST_PYTHON_FALLBACK" in text
    assert "Requested Python" in text


def test_vast_search_uses_sdk_json_and_parser_fallback() -> None:
    text = (SCRIPTS / "vast_search.sh").read_text()
    assert "VastAI(api_key=api_key).search_offers" in text
    assert "vast_parse_offers.py" in text
    assert '--limit "$limit"' in text


def test_vast_create_uses_sdk_and_preserves_safety_guards() -> None:
    shell_text = (SCRIPTS / "vast_create.sh").read_text()
    helper_text = (SCRIPTS / "vast_create_instance.py").read_text()
    assert "vast_create_instance.py" in shell_text
    assert "vast create instance" not in shell_text
    assert 'validate_price "$max_price"' in shell_text
    assert "require_yes" in shell_text
    assert "client.create_instance" in helper_text
    assert "--offer-type" in shell_text
    assert "price = None" in helper_text
    assert "price = args.bid_price if args.bid_price is not None else args.max_price" in helper_text
    assert "price=price" in helper_text
    assert "VAST_API_KEY" in helper_text


def test_vast_stop_destroy_uses_sdk_and_preserves_safety_guards() -> None:
    text = (SCRIPTS / "vast_stop_destroy.sh").read_text()
    assert "ensure_vast_cli" not in text
    assert "vast destroy instance" not in text
    assert "vast stop instance" not in text
    assert "client.destroy_instance" in text
    assert "client.stop_instance" in text
    assert 'VastAI(api_key=os.environ["VAST_API_KEY"])' in text
    assert "require_yes" in text


def test_vast_upload_and_pull_use_sdk_for_ssh_url() -> None:
    helper_text = (SCRIPTS / "vast_instance_info.py").read_text()
    common_text = (SCRIPTS / "vast_common.sh").read_text()
    assert "client.ssh_url" in helper_text
    assert "client.show_instance" in helper_text
    assert "VAST_SSH_KEY:=$HOME/.ssh/id_ed25519" in common_text
    assert "IdentitiesOnly=yes" in common_text
    for name in ["vast_run_upload.sh", "vast_pull.sh"]:
        text = (SCRIPTS / name).read_text()
        assert 'vast_instance_info.py" ssh-url' in text
        assert "vast ssh-url" not in text
        assert 'build_ssh_opts "$port"' in text


def test_existing_instance_sweep_script_uses_sweep_runner_without_create_or_destroy() -> None:
    text = (SCRIPTS / "vast_run_sweep_existing.sh").read_text()
    assert "python -m flower.sweep" in text
    assert "--instance-id" in text
    assert "vast_run_upload.sh" in text
    assert "vast_create.sh" not in text
    assert "vast_stop_destroy.sh" not in text
    assert "destroy" not in text


def test_tensorboard_script_uses_existing_instance_and_ssh_tunnel() -> None:
    text = (SCRIPTS / "vast_tensorboard.sh").read_text()
    assert "--instance-id" in text
    assert "uv run tensorboard" in text
    assert 'vast_instance_info.py" ssh-url' in text
    assert "ssh -L" in text
    assert "vast_create.sh" not in text
    assert "vast_stop_destroy.sh" not in text
    assert "destroy" not in text


def test_existing_instance_sweep_forwards_dry_run_only_when_requested() -> None:
    import subprocess
    import tempfile
    import textwrap

    with tempfile.TemporaryDirectory() as tmp:
        wrapper = Path(tmp) / "bash"
        wrapper.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/bash
                set -euo pipefail
                target={SCRIPTS / "vast_run_upload.sh"}
                for ((idx = 1; idx <= $#; idx++)); do
                  if [[ ${{!idx}} == \"$target\" ]]; then
                    printf '%s\\n' \"${{@:idx}}\"
                    exit 0
                  fi
                done
                exec /usr/bin/bash \"$@\"
                """
            )
        )
        wrapper.chmod(0o755)
        env = {"PATH": f"{tmp}:/usr/bin:/bin", "HOME": str(Path.home())}

        base_cmd = [str(SCRIPTS / "vast_run_sweep_existing.sh"), "--instance-id", "123"]
        default_out = subprocess.check_output(base_cmd, env=env, text=True)
        dry_run_out = subprocess.check_output([*base_cmd, "--dry-run"], env=env, text=True)

    assert "--dry-run" not in default_out.splitlines()
    assert "--dry-run" in dry_run_out.splitlines()


def _parse_ssh_args(value: str) -> tuple[str, str, str]:
    import subprocess

    script = f"source {SCRIPTS / 'vast_common.sh'} && ssh_args_from_url {value!r}"
    out = subprocess.check_output(["bash", "-lc", script], text=True).strip()
    user, host, port = out.split()
    return user, host, port


def test_ssh_args_from_url_accepts_vast_formats() -> None:
    assert _parse_ssh_args("ssh://root@ssh5.vast.ai:19420") == ("root", "ssh5.vast.ai", "19420")
    assert _parse_ssh_args("root@ssh5.vast.ai -p 19420") == ("root", "ssh5.vast.ai", "19420")
    assert _parse_ssh_args("ssh -p 19420 root@ssh5.vast.ai") == ("root", "ssh5.vast.ai", "19420")


def test_vast_upload_and_pull_use_parsed_nondefault_port() -> None:
    common_text = (SCRIPTS / "vast_common.sh").read_text()
    assert "parsed.port" in common_text
    assert _parse_ssh_args("ssh://root@ssh5.vast.ai:19420")[2] == "19420"
    for name in ["vast_run_upload.sh", "vast_pull.sh"]:
        text = (SCRIPTS / name).read_text()
        assert 'read -r user host port < <(ssh_args_from_url "$ssh_url")' in text
        assert 'build_ssh_opts "$port"' in text


def test_vast_run_upload_groups_remote_background_pid_write() -> None:
    text = (SCRIPTS / "vast_run_upload.sh").read_text()
    assert "printf -v remote_cmd_q '%q' \"$remote_cmd\"" in text
    assert 'ssh "${ssh_opts[@]}" "$user@$host" "bash -s -- $remote_dir_q $run_dir_q $remote_cmd_q"' in text
    assert 'cd "$remote_dir"' in text
    assert 'mkdir -p "$run_dir"' in text
    assert 'nohup bash -lc "$remote_cmd" > "$run_dir/remote.log" 2>&1 &' in text
    assert 'printf \'%s\\n\' "$!" > "$run_dir/remote.pid"' in text
    assert "& echo \\$! > '$VAST_RUN_DIR/remote.pid'" not in text


def test_benchmark_image_never_exceeds_cuda_capability() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("vast_benchmark_image", SCRIPTS / "vast_benchmark_image.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "cuda12.1" in module.select_image(12.2)
    assert "cuda12.4" in module.select_image(12.4)
    assert "cuda12.8" in module.select_image(12.8)
    assert "cuda12.9" not in module.select_image(12.8)


def test_vast_default_image_uses_cuda128_for_broad_driver_compatibility() -> None:
    text = (ROOT / "configs" / "vast_defaults.env").read_text()
    assert "VAST_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel" in text
    assert "VAST_IMAGE=pytorch/pytorch:2.8.0-cuda12.9" not in text


def test_vast_offer_parser_accepts_json_payload() -> None:
    rows = vast_parse_offers.parse_offers(
        '{"offers":[{"id":1,"gpu_name":"RTX 4090","dph_total":0.19,"reliability":0.99,"inet_up":50,"inet_down":100}]}'
    )
    assert vast_parse_offers.normalize_offer(rows[0]) == {
        "id": 1,
        "gpu": "RTX 4090",
        "num_gpus": None,
        "dph_total": 0.19,
        "reliability": 0.99,
        "inet_up": 50,
        "inet_down": 100,
    }


def test_vast_offer_parser_accepts_table_payload() -> None:
    payload = """
ID  GPU        $/hr  REL   UP  DOWN
7   RTX 3090   0.12  0.98  25  150
"""
    rows = vast_parse_offers.parse_offers(payload)
    assert vast_parse_offers.normalize_offer(rows[0])["id"] == "7"
    assert vast_parse_offers.normalize_offer(rows[0])["gpu"] == "RTX 3090"
