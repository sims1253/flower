"""Shared-memory (/dev/shm) watchdog.

FineWeb DataLoader workers + pinned memory write batches through `/dev/shm`. On
this box `/dev/shm` is only ~16 GB, and if it fills the run does not OOM cleanly —
it silently spills and throughput collapses (the "shared-memory spill" seen
throughout the Sweep 7 Phase A report). VRAM is not the binding limit; shm is.

This module runs a background thread that polls `/dev/shm` usage and aborts the
run loudly if it crosses a hard threshold, so a leak fails fast and visibly
instead of quietly destroying performance. Defaults are hardcoded (no config
surface): warn at 70%, abort at 85% of total `/dev/shm`.
"""

from __future__ import annotations

import os
import shutil
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
                gib = total / 1024**3
                print(
                    f"\n[shm-guard] FATAL: {self.path} at {frac:.0%} "
                    f"({used / 1024**3:.1f}/{gib:.1f} GiB) >= abort threshold "
                    f"{self.abort_fraction:.0%}. Aborting to avoid a silent spill that "
                    f"kills throughput. Reduce DataLoader num_workers/prefetch or batch size.",
                    flush=True,
                )
                # SIGKILL the whole process group: a leaked shm segment is usually
                # held by stuck DataLoader workers that won't respond to a soft
                # interrupt. Hard-kill guarantees the run stops instead of crawling.
                os._exit(137)
            if frac >= self.warn_fraction and not self._warned:
                self._warned = True
                print(
                    f"\n[shm-guard] WARNING: {self.path} at {frac:.0%} "
                    f"(>= {self.warn_fraction:.0%}). Approaching the spill threshold "
                    f"({self.abort_fraction:.0%} aborts).",
                    flush=True,
                )

    def start(self) -> ShmWatchdog:
        if _shm_usage(self.path) is None:
            return self  # nothing to watch
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
