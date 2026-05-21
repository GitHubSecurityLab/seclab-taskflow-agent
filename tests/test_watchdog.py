# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the process watchdog helper."""

from __future__ import annotations

import time

import pytest

from seclab_taskflow_agent import _watchdog


def test_ping_updates_timestamp():
    before = _watchdog._last_activity
    time.sleep(0.01)
    _watchdog.watchdog_ping()
    assert _watchdog._last_activity > before


def test_start_watchdog_is_idempotent():
    _watchdog._started = False
    _watchdog.start_watchdog(timeout=3600)
    first = _watchdog._started
    _watchdog.start_watchdog(timeout=3600)
    assert first is True
    assert _watchdog._started is True


def test_watchdog_loop_force_exits_when_idle(monkeypatch):
    """When no ping arrives within *timeout*, the loop must call os._exit."""
    exits: list[int] = []

    def _fake_exit(code: int) -> None:
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(_watchdog.os, "_exit", _fake_exit)
    # Simulate a stale timestamp so the first iteration decides to exit.
    monkeypatch.setattr(_watchdog, "_last_activity", time.monotonic() - 10_000)

    with pytest.raises(SystemExit) as excinfo:
        _watchdog._watchdog_loop(timeout=1)
    assert excinfo.value.code == 2
    assert exits == [2]
