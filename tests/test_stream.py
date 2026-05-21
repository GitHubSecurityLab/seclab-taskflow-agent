# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the runner stream helpers."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from seclab_taskflow_agent._stream import (
    bridge_copilot_tool_event,
    drive_backend_stream,
)
from seclab_taskflow_agent.sdk import TextDelta, ToolEnd
from seclab_taskflow_agent.sdk.errors import (
    BackendRateLimitError,
    BackendTimeoutError,
)


class _RecordingHooks:
    def __init__(self) -> None:
        self.starts: list[tuple[Any, Any, Any]] = []
        self.ends: list[tuple[Any, Any, Any, Any]] = []

    async def on_tool_start(self, ctx: Any, agent: Any, tool: Any) -> None:
        self.starts.append((ctx, agent, tool))

    async def on_tool_end(self, ctx: Any, agent: Any, tool: Any, result: Any) -> None:
        self.ends.append((ctx, agent, tool, result))


def test_bridge_emits_envelope_and_name():
    hooks = _RecordingHooks()
    asyncio.run(bridge_copilot_tool_event(ToolEnd(tool_name="echo", text="hi"), hooks))
    assert len(hooks.starts) == 1
    assert hooks.starts[0][2].name == "echo"
    assert len(hooks.ends) == 1
    _, _, tool, payload = hooks.ends[0]
    assert tool.name == "echo"
    assert json.loads(payload) == {"text": "hi"}


def test_bridge_no_hooks_is_noop():
    asyncio.run(bridge_copilot_tool_event(ToolEnd(tool_name="echo", text="hi"), None))


class _ScriptedBackend:
    def __init__(self, scripts: list[list[Any] | Exception]) -> None:
        self._scripts = scripts
        self.calls = 0

    async def run_streamed(self, _agent: Any, _prompt: str, *, max_turns: int) -> Any:
        del max_turns
        script = self._scripts[self.calls]
        self.calls += 1
        if isinstance(script, Exception):
            raise script
        for ev in script:
            yield ev


def _drive(backend: _ScriptedBackend, hooks: Any = None, **kwargs: Any) -> None:
    asyncio.run(
        drive_backend_stream(
            backend_impl=backend,
            agent_handle=None,
            prompt="go",
            max_turns=10,
            run_hooks=hooks,
            async_task=False,
            task_id="t",
            max_api_retry=kwargs.get("max_api_retry", 1),
            initial_rate_limit_backoff=kwargs.get("initial_rate_limit_backoff", 1),
            max_rate_limit_backoff=kwargs.get("max_rate_limit_backoff", 4),
        )
    )


def test_drive_renders_text_and_forwards_tool_event(monkeypatch):
    rendered: list[str] = []

    async def _fake_render(text: str, **_kw: Any) -> None:
        rendered.append(text)

    monkeypatch.setattr("seclab_taskflow_agent._stream.render_model_output", _fake_render)
    hooks = _RecordingHooks()
    backend = _ScriptedBackend([[TextDelta(text="hi"), ToolEnd(tool_name="echo", text="r")]])

    _drive(backend, hooks)

    assert "hi" in rendered
    assert hooks.ends
    assert json.loads(hooks.ends[0][3]) == {"text": "r"}


def test_drive_retries_then_succeeds_on_timeout(monkeypatch):
    monkeypatch.setattr(
        "seclab_taskflow_agent._stream.render_model_output",
        lambda *_a, **_kw: _noop(),
    )
    backend = _ScriptedBackend([BackendTimeoutError("once"), [TextDelta(text="ok")]])
    _drive(backend, max_api_retry=2)
    assert backend.calls == 2


def test_drive_raises_after_retries_exhausted(monkeypatch):
    monkeypatch.setattr(
        "seclab_taskflow_agent._stream.render_model_output",
        lambda *_a, **_kw: _noop(),
    )
    backend = _ScriptedBackend([BackendTimeoutError("a"), BackendTimeoutError("b")])
    with pytest.raises(BackendTimeoutError):
        _drive(backend, max_api_retry=1)


def test_drive_caps_rate_limit_backoff(monkeypatch):
    monkeypatch.setattr(
        "seclab_taskflow_agent._stream.render_model_output",
        lambda *_a, **_kw: _noop(),
    )
    sleeps: list[float] = []

    async def _fake_sleep(n: float) -> None:
        sleeps.append(n)

    monkeypatch.setattr("seclab_taskflow_agent._stream.asyncio.sleep", _fake_sleep)
    backend = _ScriptedBackend(
        [
            BackendRateLimitError("rl1"),
            BackendRateLimitError("rl2"),
            [TextDelta(text="done")],
        ]
    )
    _drive(backend, initial_rate_limit_backoff=1, max_rate_limit_backoff=4)
    assert sleeps
    assert all(s <= 4 for s in sleeps)


async def _noop() -> None:
    return None


class _HangingBackend:
    """A backend whose stream yields once then blocks forever — used to
    exercise the idle-timeout path of drive_backend_stream.
    """

    async def run_streamed(self, _agent: Any, _prompt: str, *, max_turns: int) -> Any:
        del max_turns
        yield TextDelta(text="first")
        await asyncio.Event().wait()  # hang


def test_drive_raises_on_stream_idle_timeout(monkeypatch):
    monkeypatch.setattr(
        "seclab_taskflow_agent._stream.render_model_output",
        lambda *_a, **_kw: _noop(),
    )
    # Force a tiny idle timeout so the test runs quickly.
    monkeypatch.setattr("seclab_taskflow_agent._stream.STREAM_IDLE_TIMEOUT", 0.05)

    backend = _HangingBackend()
    with pytest.raises(BackendTimeoutError, match="idle"):
        asyncio.run(
            drive_backend_stream(
                backend_impl=backend,
                agent_handle=None,
                prompt="p",
                max_turns=1,
                run_hooks=None,
                async_task=False,
                task_id="t",
                max_api_retry=0,
                initial_rate_limit_backoff=1,
                max_rate_limit_backoff=4,
            )
        )


def test_drive_pings_watchdog_per_event(monkeypatch):
    monkeypatch.setattr(
        "seclab_taskflow_agent._stream.render_model_output",
        lambda *_a, **_kw: _noop(),
    )
    pings: list[int] = []
    monkeypatch.setattr(
        "seclab_taskflow_agent._stream.watchdog_ping",
        lambda: pings.append(1),
    )
    backend = _ScriptedBackend(
        [[TextDelta(text="a"), TextDelta(text="b"), ToolEnd(tool_name="t", text="x")]]
    )
    _drive(backend, _RecordingHooks())
    assert len(pings) == 3
