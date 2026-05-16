#!/usr/bin/env python3
"""Launch TensorBoard with a timeout around its localhost port probe.

On some WSL2 setups, connecting to an unused localhost port can hang instead of
returning ECONNREFUSED immediately. TensorBoard checks whether the requested port
is in use with such a connect during startup, so it appears to freeze before it
prints the URL. This wrapper makes that probe time out quickly.
"""

from __future__ import annotations

import socket

_real_socket = socket.socket


class TimeoutSocket:
    def __init__(self, *args, **kwargs):
        self._socket = _real_socket(*args, **kwargs)

    def __enter__(self):
        self._socket.__enter__()
        return self

    def __exit__(self, *args):
        return self._socket.__exit__(*args)

    def connect_ex(self, address):
        old_timeout = self._socket.gettimeout()
        self._socket.settimeout(0.05)
        try:
            return self._socket.connect_ex(address)
        except TimeoutError:
            return 111
        finally:
            self._socket.settimeout(old_timeout)

    def __getattr__(self, name):
        return getattr(self._socket, name)


def main() -> int:
    socket.socket = TimeoutSocket
    from tensorboard.main import run_main

    return int(run_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
