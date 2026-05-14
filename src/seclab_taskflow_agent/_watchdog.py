# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Process-level watchdog that force-exits if the event loop stops progressing.

The asyncio retry loop, the httpx client timeouts, and the per-stream
idle timeout already cover the cases we know how to recover from. This
module is the last-resort backstop for everything else (a stuck MCP
cleanup, an asyncio loop spinning on a leaked task, a kernel-level
socket pathology) — a daemon thread polls a monotonic timestamp that
the runtime updates from every interesting event and force-exits the
process if the timestamp ever goes stale for too long.

Sources of pings:

* :func:`drive_backend_stream` — every backend event.
* The runner's ``on_tool_start`` / ``on_tool_end`` hooks.
* The runner's MCP cleanup / backend ``aclose`` paths.

The default timeout is intentionally larger than every recoverable
timeout below it so the watchdog never fires before the asyncio layer
has had a chance to recover.
"""

from __future__ import annotations

__all__ = ["WATCHDOG_IDLE_TIMEOUT", "start_watchdog", "watchdog_ping"]

import logging
import os
import sys
import threading
import time

# 35 minutes by default — comfortably above the per-stream idle timeout
# (30 min) and the rate-limit backoff cap (2 min) so the watchdog only
# trips on hangs the asyncio path could not recover from.
WATCHDOG_IDLE_TIMEOUT = int(os.environ.get("WATCHDOG_IDLE_TIMEOUT", "2100"))

_last_activity = time.monotonic()
_lock = threading.Lock()
_started = False


def watchdog_ping() -> None:
    """Record activity. Safe to call from any coroutine or callback."""
    global _last_activity
    with _lock:
        _last_activity = time.monotonic()


def _watchdog_loop(timeout: int) -> None:
    check_interval = min(60, max(1, timeout // 5))
    while True:
        time.sleep(check_interval)
        with _lock:
            idle = time.monotonic() - _last_activity
        if idle > timeout:
            logging.error(
                "Watchdog: no activity for %.0fs (limit %ds) — force-exiting to prevent hang",
                idle,
                timeout,
            )
            sys.stderr.flush()
            sys.stdout.flush()
            os._exit(2)


def start_watchdog(timeout: int = WATCHDOG_IDLE_TIMEOUT) -> None:
    """Start the watchdog thread once per process (idempotent)."""
    global _started
    if _started:
        return
    _started = True
    watchdog_ping()  # reset timestamp so a late call doesn't trip immediately
    threading.Thread(
        target=_watchdog_loop, args=(timeout,), daemon=True, name="seclab-watchdog"
    ).start()
