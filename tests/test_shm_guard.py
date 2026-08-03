from __future__ import annotations

import time

from flower import shm_guard
from flower.shm_guard import ShmWatchdog, start_shm_watchdog


def test_watchdog_warns_then_aborts(monkeypatch) -> None:
    # Simulate /dev/shm climbing past warn (0.70) then abort (0.85).
    fractions = iter([0.50, 0.75, 0.90])

    def fake_usage(path: str = "/dev/shm"):
        try:
            frac = next(fractions)
        except StopIteration:
            frac = 0.90
        total = 16 * 1024**3
        return frac * total, total

    aborted: dict[str, int] = {}

    def fake_exit(code: int) -> None:  # stand in for os._exit
        aborted["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr(shm_guard, "_shm_usage", fake_usage)
    monkeypatch.setattr(shm_guard.os, "_exit", fake_exit)

    wd = ShmWatchdog(poll_seconds=0.01)
    # Drive the loop synchronously instead of via the daemon thread.
    try:
        wd._loop()
    except SystemExit:
        pass

    assert aborted.get("code") == 137
    assert wd._warned is True
    assert wd.peak_fraction >= 0.85


def test_watchdog_noop_when_no_shm(monkeypatch) -> None:
    monkeypatch.setattr(shm_guard, "_shm_usage", lambda path="/dev/shm": None)
    wd = start_shm_watchdog()  # should not start a thread or raise
    assert wd._thread is None
    wd.stop()


def test_watchdog_start_stop_clean(monkeypatch) -> None:
    # A healthy shm (low usage) should start, poll, and stop without aborting.
    monkeypatch.setattr(shm_guard, "_shm_usage", lambda path="/dev/shm": (1.0 * 1024**3, 16.0 * 1024**3))
    wd = ShmWatchdog(poll_seconds=0.01).start()
    time.sleep(0.05)
    wd.stop()
    assert wd._warned is False
