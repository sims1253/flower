"""Shared-memory (/dev/shm) watchdog.

FineWeb DataLoader workers + pinned memory write batches through `/dev/shm`. On
this box `/dev/shm` is only ~16 GB, and if it fills the run does not OOM cleanly —
it silently spills and throughput collapses (the "shared-memory spill" seen
throughout the Sweep 7 Phase A report). VRAM is not the binding limit; shm is.

This module runs a background thread that polls `/dev/shm` usage and aborts the
run loudly if it crosses a hard threshold, so a leak fails fast and visibly
instead of quietly destroying performance. Defaults are hardcoded (no config
surface): warn at 70%, abort at 85% of total `/dev/shm`.

The abort SIGKILLs the run's whole process group (claimed at startup via
`claim_process_group`), because the interesting /dev/shm memory is held by
forked DataLoader workers that a bare `os._exit` would orphan.
"""

from __future__ import annotations

import os
import shutil
import signal
import threading

# Hardcoded safe defaults.
_WARN_FRACTION = 0.70
_ABORT_FRACTION = 0.85
_POLL_SECONDS = 2.0
_SHM_PATH = "/dev/shm"


def _shm_usage(path: str = _SHM_PATH) -> tuple[float, float] | None:
    """Return (used_bytes, total_bytes) for the shm filesystem, or None if absent."""
    try:
        usage = shutil.disk_usage(path)
    except (FileNotFoundError, PermissionError, OSError):
        return None
    return float(usage.used), float(usage.total)


def claim_process_group() -> bool:
    """Move this process into its own process group: best-effort `os.setpgid(0, 0)`.

    The abort path can only clean up DataLoader workers if they sit in OUR
    process group, and workers inherit the group at fork time — so the claim
    must happen before any worker is spawned. train.py triggers it from the
    resume fast-forward (the first DataLoader consumer on a resumed run, which
    runs before the watchdog itself starts) and from `ShmWatchdog.start`
    (before the training loop's first pull); both call sites are idempotent.

    Fails (returning False) only where `setpgid` is forbidden, e.g. EPERM when
    the caller is already a session leader (a `setsid`-style launcher). That
    case is still safe for the kill path: a session leader normally leads its
    own process group, which is the property `kill_own_process_group` checks.
    """
    try:
        os.setpgid(0, 0)
    except (AttributeError, OSError):
        # AttributeError: platforms without setpgid at all (native Windows) —
        # the claim is best-effort, never a crash (train.py calls it
        # unconditionally on resume). OSError: forbidden, e.g. EPERM when the
        # caller is already a session leader.
        return False
    return True


def kill_own_process_group() -> bool:
    """SIGKILL this process's group iff this process leads it. True = kill ordered.

    Why killpg instead of the old bare `os._exit(137)`: a plain exit skips
    nothing on purpose but kills nothing either — the DataLoader's persistent
    daemon workers are separate processes, and each survivor keeps pinning its
    /dev/shm tensors. The "fail fast" abort therefore used to leave the machine
    in exactly the state that trips the NEXT run's threshold the moment it
    starts. SIGKILL because stuck workers do not respond to a soft interrupt.

    Safety gate: only fires when `os.getpgrp() == os.getpid()`. A group
    leader's group contains exactly itself and its forked descendants (the
    workers), so the killpg cannot reach an unrelated process. If the startup
    claim failed and we still sit in the CALLER's group, killpg would take out
    a shell / test harness / whole CI job with us — so we decline (return
    False) and the caller falls back to exiting this process alone.
    """
    if os.getpgrp() != os.getpid():
        return False
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except OSError:
        return False
    return True  # unreachable in practice: SIGKILL to our own group does not return


class ShmWatchdog:
    """Background poller that aborts the process if /dev/shm fills past a threshold."""

    def __init__(
        self,
        *,
        warn_fraction: float = _WARN_FRACTION,
        abort_fraction: float = _ABORT_FRACTION,
        poll_seconds: float = _POLL_SECONDS,
        path: str = _SHM_PATH,
    ) -> None:
        self.warn_fraction = warn_fraction
        self.abort_fraction = abort_fraction
        self.poll_seconds = poll_seconds
        self.path = path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False
        self.peak_fraction = 0.0

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            usage = _shm_usage(self.path)
            if usage is None:
                return  # no shm filesystem (e.g. macOS / container without it) — nothing to guard
            used, total = usage
            if total <= 0:
                continue
            frac = used / total
            self.peak_fraction = max(self.peak_fraction, frac)
            if frac >= self.abort_fraction:
                self._abort(used, total, frac)
                return  # the abort killed the process; never poll again
            if frac >= self.warn_fraction and not self._warned:
                self._warned = True
                print(
                    f"\n[shm-guard] WARNING: {self.path} at {frac:.0%} "
                    f"(>= {self.warn_fraction:.0%}). Approaching the spill threshold "
                    f"({self.abort_fraction:.0%} aborts).",
                    flush=True,
                )

    def _abort(self, used: float, total: float, frac: float) -> None:
        gib = total / 1024**3
        # peak_fraction is printed BEFORE dying so the run's high-water mark
        # lands in the logs even though the process is about to be SIGKILLed
        # (SIGKILL precludes any atexit/finally logging after this point).
        print(
            f"\n[shm-guard] FATAL: {self.path} at {frac:.0%} "
            f"({used / 1024**3:.1f}/{gib:.1f} GiB) >= abort threshold "
            f"{self.abort_fraction:.0%}; peak this run {self.peak_fraction:.0%}. "
            f"Aborting to avoid a silent spill that kills throughput. Reduce "
            f"DataLoader num_workers/prefetch or batch size.",
            flush=True,
        )
        # Kill the whole process group: the leaked shm is held by the DataLoader
        # workers, and a bare os._exit(137) here used to orphan them — each
        # survivor kept pinning its shm tensors, so the next run tripped the
        # same threshold immediately on startup.
        if kill_own_process_group():
            return  # unreachable in production: the group kill takes this thread too
        print(
            "[shm-guard] process-group kill unavailable (this process is not its "
            "group leader — the startup setpgid failed); exiting this process "
            "only. Orphaned DataLoader workers may still hold /dev/shm; kill "
            "them manually before the next run.",
            flush=True,
        )
        os._exit(137)

    def start(self) -> ShmWatchdog:
        if _shm_usage(self.path) is None:
            return self  # nothing to watch
        # Claim a dedicated process group BEFORE any DataLoader worker forks
        # (workers inherit the group; the abort path kills by group). If this
        # fails, _abort degrades to a lonely os._exit rather than risking the
        # caller's group. Idempotent; train.py also claims earlier on resume
        # runs where the fast-forward consumes the DataLoader first.
        claim_process_group()
        self._thread = threading.Thread(target=self._loop, name="shm-guard", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_seconds + 1.0)


def start_shm_watchdog() -> ShmWatchdog:
    """Start the default /dev/shm watchdog. Safe no-op where /dev/shm is absent."""
    return ShmWatchdog().start()
