from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from flower import shm_guard
from flower.shm_guard import ShmWatchdog, start_shm_watchdog


def test_watchdog_warns_then_aborts(monkeypatch, capsys) -> None:
    # Simulate /dev/shm climbing past warn (0.70) then abort (0.85).
    fractions = iter([0.50, 0.75, 0.90])

    def fake_usage(path: str = "/dev/shm"):
        try:
            frac = next(fractions)
        except StopIteration:
            frac = 0.90
        total = 16 * 1024**3
        return frac * total, total

    killed: dict[str, bool] = {}

    def fake_kill() -> bool:  # stand in for the group SIGKILL
        killed["group"] = True
        raise SystemExit(137)

    monkeypatch.setattr(shm_guard, "_shm_usage", fake_usage)
    # NEVER let the real killpg run under pytest: the harness may itself be a
    # process-group leader, in which case the real path would SIGKILL it.
    monkeypatch.setattr(shm_guard, "kill_own_process_group", fake_kill)

    wd = ShmWatchdog(poll_seconds=0.01)
    # Drive the loop synchronously instead of via the daemon thread.
    try:
        wd._loop()
    except SystemExit:
        pass

    assert killed.get("group") is True
    assert wd._warned is True
    assert wd.peak_fraction >= 0.85
    # The run's peak must land in the logs before the process dies.
    assert "peak this run 90%" in capsys.readouterr().out


def test_watchdog_abort_falls_back_to_exit_when_not_group_leader(monkeypatch, capsys) -> None:
    """If the process-group kill is unavailable (not a leader), the watchdog
    must still die loudly via os._exit instead of hanging or polling on."""
    monkeypatch.setattr(shm_guard, "_shm_usage", lambda path="/dev/shm": (0.95 * 16 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(shm_guard, "kill_own_process_group", lambda: False)

    exited: dict[str, int] = {}

    def fake_exit(code: int) -> None:
        exited["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr(shm_guard.os, "_exit", fake_exit)

    wd = ShmWatchdog(poll_seconds=0.01)
    with pytest.raises(SystemExit):
        wd._loop()

    assert exited.get("code") == 137
    out = capsys.readouterr().out
    assert "not its group leader" in out  # the operator learns workers may survive


def test_kill_own_process_group_gated_on_leadership(monkeypatch) -> None:
    """killpg may only fire when this process LEADS its group; otherwise the
    kill would reach the caller's shell/harness."""
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(shm_guard.os, "getpid", lambda: 100)
    monkeypatch.setattr(
        shm_guard.os, "killpg", lambda pgid, sig: calls.append((pgid, sig))
    )

    # Leader: pgid == pid -> killpg(SIGKILL) fires.
    monkeypatch.setattr(shm_guard.os, "getpgrp", lambda: 100)
    assert shm_guard.kill_own_process_group() is True
    assert calls == [(100, signal.SIGKILL)]

    # Not leader (inherited the caller's group) -> decline, kill nothing.
    calls.clear()
    monkeypatch.setattr(shm_guard.os, "getpgrp", lambda: 42)
    assert shm_guard.kill_own_process_group() is False
    assert calls == []


def test_kill_own_process_group_swallows_killpg_errors(monkeypatch) -> None:
    monkeypatch.setattr(shm_guard.os, "getpid", lambda: 100)
    monkeypatch.setattr(shm_guard.os, "getpgrp", lambda: 100)

    def boom(pgid, sig):
        raise ProcessLookupError("group already gone")

    monkeypatch.setattr(shm_guard.os, "killpg", boom)
    assert shm_guard.kill_own_process_group() is False


def test_claim_process_group_is_best_effort(monkeypatch) -> None:
    setpgid_calls: list[tuple[int, int]] = []

    def fake_setpgid(pid, pgid):
        setpgid_calls.append((pid, pgid))

    monkeypatch.setattr(shm_guard.os, "setpgid", fake_setpgid)
    assert shm_guard.claim_process_group() is True
    assert setpgid_calls == [(0, 0)]

    # EPERM (e.g. already a session leader) must not raise — it degrades the
    # abort path, it does not crash the run.
    def deny(pid, pgid):
        raise PermissionError("session leader")

    monkeypatch.setattr(shm_guard.os, "setpgid", deny)
    assert shm_guard.claim_process_group() is False


def test_watchdog_start_claims_process_group(monkeypatch) -> None:
    """start() must claim the group BEFORE workers can fork."""
    claimed: list[bool] = []
    monkeypatch.setattr(shm_guard, "claim_process_group", lambda: claimed.append(True) or True)
    monkeypatch.setattr(shm_guard, "_shm_usage", lambda path="/dev/shm": (1.0 * 1024**3, 16.0 * 1024**3))

    wd = ShmWatchdog(poll_seconds=0.01).start()
    try:
        assert claimed == [True]
    finally:
        wd.stop()


def test_watchdog_noop_when_no_shm(monkeypatch) -> None:
    monkeypatch.setattr(shm_guard, "_shm_usage", lambda path="/dev/shm": None)
    wd = start_shm_watchdog()  # should not start a thread or raise
    assert wd._thread is None
    wd.stop()


def test_watchdog_start_stop_clean(monkeypatch) -> None:
    # A healthy shm (low usage) should start, poll, and stop without aborting.
    monkeypatch.setattr(shm_guard, "claim_process_group", lambda: True)  # no real setpgid under pytest
    monkeypatch.setattr(shm_guard, "_shm_usage", lambda path="/dev/shm": (1.0 * 1024**3, 16.0 * 1024**3))
    wd = ShmWatchdog(poll_seconds=0.01).start()
    time.sleep(0.05)
    wd.stop()
    assert wd._warned is False


# End-to-end proof of the real kill path. Runs in a SUBPROCESS so the group
# SIGKILL can never touch the pytest harness. The child mimics a DataLoader
# daemon worker: forked after the group claim (so it inherits the pgid) and
# heartbeat-writing; if the abort's killpg reaches it, the heartbeats stop.
_CHILD_SCRIPT = """
import os, sys, time
sys.path.insert(0, {repo!r})
from flower.shm_guard import claim_process_group, kill_own_process_group

child_file = sys.argv[1]
claim_process_group()
pid = os.fork()
if pid == 0:  # the "DataLoader worker": inherited our process group
    n = 0
    while True:
        n += 1
        with open(child_file, "w") as f:
            f.write(str(n))
        time.sleep(0.05)
# parent: wait until the worker proves it runs, then abort like the watchdog does
deadline = time.time() + 10
while not os.path.exists(child_file):
    if time.time() > deadline:
        print("WORKER NEVER STARTED", flush=True)
        os._exit(98)
    time.sleep(0.01)
time.sleep(0.2)
kill_own_process_group()
print("SURVIVED OWN GROUP KILL", flush=True)  # must never happen
os._exit(99)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork (POSIX)")
def test_abort_kills_forked_workers_in_the_process_group(tmp_path):
    repo = str(Path(shm_guard.__file__).resolve().parents[1])
    heartbeat = tmp_path / "worker_heartbeat.txt"
    script = _CHILD_SCRIPT.format(repo=repo)

    # start_new_session=False matters: the child must inherit pytest's group so
    # a bug in the leadership gate would be caught by CI dying, not hidden.
    proc = subprocess.run(
        [sys.executable, "-c", script, str(heartbeat)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=repo,
    )

    assert "SURVIVED OWN GROUP KILL" not in proc.stdout
    # The parent was SIGKILLed by its own killpg (negative returncode = signal).
    assert proc.returncode == -signal.SIGKILL, proc.stdout + proc.stderr

    # The forked "worker" must have died with the group: its heartbeat freezes.
    first = heartbeat.read_text()
    time.sleep(0.4)
    assert heartbeat.read_text() == first, "worker kept running after the group kill"
