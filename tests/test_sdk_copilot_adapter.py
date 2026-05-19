# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Copilot SDK adapter glue."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("copilot")

from copilot.generated.session_events import SessionEventType

from seclab_taskflow_agent.sdk import get_backend
from seclab_taskflow_agent.sdk.base import AgentSpec, MCPServerSpec, TextDelta
from seclab_taskflow_agent.sdk.copilot_sdk.backend import (
    CopilotSDKBackend,
    _reasoning_effort,
    _normalize_model,
)
from seclab_taskflow_agent.sdk.errors import (
    BackendBadRequestError,
    BackendCapabilityError,
    BackendUnexpectedError,
)


def test_get_backend_returns_copilot_sdk_instance():
    backend = get_backend("copilot_sdk")
    assert isinstance(backend, CopilotSDKBackend)
    assert backend.name == "copilot_sdk"


def _spec(**overrides) -> AgentSpec:
    base = {
        "name": "a",
        "instructions": "",
        "model": "gpt-5-mini",
    }
    base.update(overrides)
    return AgentSpec(**base)


def test_validate_accepts_minimal_spec():
    CopilotSDKBackend().validate(_spec())


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"in_handoff_graph": True}, "handoffs"),
        ({"model_settings": {"temperature": 0.0}}, "temperature"),
        ({"model_settings": {"parallel_tool_calls": True}}, "parallel_tool_calls"),
    ],
)
def test_validate_rejects_unsupported(kwargs, field):
    backend = CopilotSDKBackend()
    with pytest.raises(BackendCapabilityError, match=field):
        backend.validate(_spec(**kwargs))


def test_validate_accepts_exclude_from_context():
    CopilotSDKBackend().validate(_spec(exclude_from_context=True))


def test_validate_rejects_handoff_targets():
    backend = CopilotSDKBackend()
    with pytest.raises(BackendCapabilityError, match="handoffs"):
        backend.validate(_spec(handoffs=[_spec(name="b")]))


@dataclass
class _Event:
    type: SessionEventType
    data: Any = None


class _FakeSession:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, prompt: str) -> str:
        self.sent.append(prompt)
        return "req-1"


def test_run_streamed_translates_deltas_until_idle():
    async def _run():
        session = _FakeSession()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        for ev in (
            _Event(SessionEventType.ASSISTANT_MESSAGE_DELTA, SimpleNamespace(delta_content="hi ")),
            _Event(SessionEventType.ASSISTANT_MESSAGE_DELTA, SimpleNamespace(delta_content="there")),
            _Event(SessionEventType.SESSION_IDLE),
        ):
            queue.put_nowait(ev)
        handle = SimpleNamespace(client=None, session=session, event_queue=queue)
        backend = CopilotSDKBackend()
        return session, [
            event async for event in backend.run_streamed(handle, "go", max_turns=10)
        ]

    session, out = asyncio.run(_run())
    assert session.sent == ["go"]
    assert out == [TextDelta(text="hi "), TextDelta(text="there")]


def test_run_streamed_session_error_raises():
    async def _run():
        queue: asyncio.Queue[Any] = asyncio.Queue()
        queue.put_nowait(_Event(SessionEventType.SESSION_ERROR, SimpleNamespace(message="boom")))
        handle = SimpleNamespace(client=None, session=_FakeSession(), event_queue=queue)
        async for _ in CopilotSDKBackend().run_streamed(handle, "go", max_turns=10):
            pass

    with pytest.raises(BackendUnexpectedError, match="boom"):
        asyncio.run(_run())


def test_run_streamed_aborts_after_tool_when_excluded():
    aborted: list[bool] = []

    class _AbortableSession(_FakeSession):
        async def abort(self) -> None:
            aborted.append(True)

    async def _run():
        session = _AbortableSession()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        result = SimpleNamespace(
            contents=[SimpleNamespace(text="tool-out")], content=None
        )
        for ev in (
            _Event(
                SessionEventType.TOOL_EXECUTION_COMPLETE,
                SimpleNamespace(success=True, result=result, tool_name="echo"),
            ),
            # idle should NOT be reached because exclude_from_context aborts
            _Event(
                SessionEventType.ASSISTANT_MESSAGE_DELTA,
                SimpleNamespace(delta_content="should-not-see"),
            ),
            _Event(SessionEventType.SESSION_IDLE),
        ):
            queue.put_nowait(ev)
        handle = SimpleNamespace(
            client=None,
            session=session,
            event_queue=queue,
            exclude_from_context=True,
        )
        return [
            ev async for ev in CopilotSDKBackend().run_streamed(handle, "go", max_turns=10)
        ]

    out = asyncio.run(_run())
    assert aborted == [True]
    # Only the ToolEnd is emitted; the trailing delta is not consumed.
    assert len(out) == 1
    assert out[0].text == "tool-out"


def test_run_streamed_keeps_running_when_exclude_disabled():
    async def _run():
        queue: asyncio.Queue[Any] = asyncio.Queue()
        result = SimpleNamespace(
            contents=[SimpleNamespace(text="tool-out")], content=None
        )
        for ev in (
            _Event(
                SessionEventType.TOOL_EXECUTION_COMPLETE,
                SimpleNamespace(success=True, result=result, tool_name="echo"),
            ),
            _Event(
                SessionEventType.ASSISTANT_MESSAGE_DELTA,
                SimpleNamespace(delta_content="post-tool"),
            ),
            _Event(SessionEventType.SESSION_IDLE),
        ):
            queue.put_nowait(ev)
        handle = SimpleNamespace(
            client=None,
            session=_FakeSession(),
            event_queue=queue,
            exclude_from_context=False,
        )
        return [
            ev async for ev in CopilotSDKBackend().run_streamed(handle, "go", max_turns=10)
        ]

    out = asyncio.run(_run())
    assert [type(e).__name__ for e in out] == ["ToolEnd", "TextDelta"]


def test_run_streamed_aborts_session_on_consumer_break():
    """If the consumer abandons the iterator mid-turn, the adapter must
    still call session.abort() so the SDK background task can drain.
    """
    aborted: list[bool] = []

    class _AbortableSession(_FakeSession):
        async def abort(self) -> None:
            aborted.append(True)

    async def _run():
        queue: asyncio.Queue[Any] = asyncio.Queue()
        for ev in (
            _Event(
                SessionEventType.ASSISTANT_MESSAGE_DELTA,
                SimpleNamespace(delta_content="hello"),
            ),
            _Event(
                SessionEventType.ASSISTANT_MESSAGE_DELTA,
                SimpleNamespace(delta_content="world"),
            ),
            _Event(SessionEventType.SESSION_IDLE),
        ):
            queue.put_nowait(ev)
        handle = SimpleNamespace(
            client=None,
            session=_AbortableSession(),
            event_queue=queue,
            exclude_from_context=False,
        )
        # Take only the first event then break — exercises the finally
        # path of the async generator.
        async for _ev in CopilotSDKBackend().run_streamed(handle, "go", max_turns=10):
            break

    asyncio.run(_run())
    assert aborted == [True]


def test_invalid_reasoning_effort_rejected():
    with pytest.raises(BackendBadRequestError, match="reasoning_effort"):
        _reasoning_effort({"reasoning_effort": "ludicrous"})


def test_normalize_model_rejects_empty():
    # Passing an empty model would let the SDK silently fall back to its
    # built-in default, which would invalidate any reproducibility
    # guarantee a taskflow makes about the model under test.
    with pytest.raises(BackendBadRequestError, match="model is required"):
        _normalize_model("")


def test_normalize_model_strips_provider_prefix():
    assert _normalize_model("openai/gpt-4.1") == "gpt-4.1"
    assert _normalize_model("gpt-4.1") == "gpt-4.1"


def test_aclose_handles_none():
    asyncio.run(CopilotSDKBackend().aclose(None))


def test_aclose_swallows_disconnect_failures():
    class _BadSession:
        async def disconnect(self) -> None:
            raise RuntimeError("nope")

    class _Client:
        async def stop(self) -> None:
            return None

    handle = SimpleNamespace(client=_Client(), session=_BadSession(), event_queue=asyncio.Queue())
    asyncio.run(CopilotSDKBackend().aclose(handle))


def test_mcp_specs_are_used():
    # Sanity: MCPServerSpec is what build() consumes via build_mcp_config.
    spec = _spec(mcp_servers=[MCPServerSpec(name="m", kind="stdio", params={"command": "x"})])
    CopilotSDKBackend().validate(spec)
