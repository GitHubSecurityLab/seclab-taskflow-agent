# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Copilot SDK adapter glue."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

pytest.importorskip("copilot")

import copilot
from copilot.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    AssistantUsageData,
    SessionEvent,
    SessionEventType,
    SessionIdleData,
    UserMessageData,
)
from mcp.types import Tool

from seclab_taskflow_agent.sdk import get_backend
from seclab_taskflow_agent.sdk.base import AgentSpec, MCPServerSpec, TextDelta, TokenUsage, ToolEnd
from seclab_taskflow_agent.sdk.copilot_sdk.backend import (
    CopilotSDKBackend,
    _reasoning_effort,
    _normalize_model,
)
from seclab_taskflow_agent.sdk.errors import (
    BackendBadRequestError,
    BackendCapabilityError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnexpectedError,
)


def test_get_backend_returns_copilot_sdk_instance():
    backend = get_backend("copilot_sdk")
    assert isinstance(backend, CopilotSDKBackend)
    assert backend.name == "copilot_sdk"


def _spec(**overrides) -> AgentSpec:
    base: dict[str, Any] = {
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


MODEL = "gpt-5.6-luna"
MSG_ID = "req-1"
SECRET = "offline-test-token"  # noqa: S105 - offline fixture, not a credential
_SDK_CLIENT = copilot.CopilotClient
_CREATE_SIGNATURE = inspect.signature(copilot.CopilotClient.create_session)


@dataclass
class _Event:
    type: SessionEventType
    data: Any = None
    agent_id: str | None = None


def _ev(etype: SessionEventType, agent_id: str | None = None, **data: Any) -> _Event:
    return _Event(etype, SimpleNamespace(**data), agent_id)


def _started(message_id: str = MSG_ID) -> _Event:
    return _ev(SessionEventType.USER_MESSAGE, message_id=message_id)


def _idle(aborted: bool = False) -> _Event:
    return _ev(SessionEventType.SESSION_IDLE, aborted=aborted)


def _delta(text: str, message_id: str = "m1") -> _Event:
    return _ev(SessionEventType.ASSISTANT_MESSAGE_DELTA, delta_content=text, message_id=message_id)


class _FakeSession:
    def __init__(self, events: list[Any] | None = None, on_event: Any = None) -> None:
        self.sent: list[str] = []
        self.aborted = 0
        self.disconnected = 0
        self._events = events or []
        self._on_event = on_event
        self.send_error: BaseException | None = None

    async def send(self, prompt: str) -> str:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(prompt)
        # Simulate the SDK dispatching events from its reader thread.
        if self._on_event is not None:
            for ev in self._events:
                self._on_event(ev)
        return MSG_ID

    async def abort(self) -> None:
        self.aborted += 1

    async def disconnect(self) -> None:
        self.disconnected += 1


def _handle(
    events: list[Any],
    *,
    stale: Sequence[Any] = (),
    session: Any = None,
    exclude: bool = False,
    model: str = MODEL,
):
    """Return a handle whose session emits *events* when ``send`` is called.

    *stale* events are queued before ``send`` (i.e. left over from a
    previous turn).
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    for ev in stale:
        queue.put_nowait(ev)
    session = session or _FakeSession()
    session._events = events
    session._on_event = queue.put_nowait
    return SimpleNamespace(
        client=None,
        session=session,
        event_queue=queue,
        exclude_from_context=exclude,
        model=model,
    )


async def _collect(handle: Any, prompt: str = "go") -> list[Any]:
    async with asyncio.timeout(5):
        return [ev async for ev in CopilotSDKBackend().run_streamed(handle, prompt, max_turns=10)]


def _stream(events: list[Any], **kw) -> tuple[Any, list[Any]]:
    handle = _handle(events, **kw)
    return handle, asyncio.run(_collect(handle))


# --- run_streamed -----------------------------------------------------------


def test_run_streamed_translates_deltas_until_idle():
    prompt = "Explain CWE-476 in one sentence."
    handle = _handle([_started(), _delta("hi "), _delta("there"), _idle(), _delta("late")])
    out = asyncio.run(_collect(handle, prompt))
    assert handle.session.sent == [prompt]
    assert out == [TextDelta(text="hi "), TextDelta(text="there")]
    assert handle.session.aborted == 0


def test_run_streamed_ignores_events_before_own_user_message():
    # A late idle/delta from an earlier (aborted) turn must not end this one.
    _, out = _stream([_delta("stale"), _idle(aborted=True), _started("other"), _started(), _delta("fresh"), _idle()])
    assert out == [TextDelta(text="fresh")]


def test_run_streamed_drains_stale_queue_before_send():
    _, out = _stream([_started(), _delta("new"), _idle()], stale=[_started(), _delta("old"), _idle()])
    assert out == [TextDelta(text="new")]


def test_run_streamed_falls_back_to_full_assistant_message():
    _, out = _stream(
        [
            _started(),
            _ev(SessionEventType.ASSISTANT_MESSAGE, content="whole", message_id="m1", model=MODEL),
            _idle(),
        ]
    )
    assert out == [TextDelta(text="whole")]


def test_run_streamed_does_not_duplicate_streamed_message():
    _, out = _stream(
        [
            _started(),
            _delta("part", message_id="m1"),
            _ev(SessionEventType.ASSISTANT_MESSAGE, content="part", message_id="m1", model=MODEL),
            _idle(),
        ]
    )
    assert out == [TextDelta(text="part")]


def test_streaming_with_public_sdk_event_types():
    def event(etype, data):
        return SessionEvent(type=etype, data=data, id=uuid4(), timestamp=datetime.now(timezone.utc))

    _, out = _stream(
        [
            event(SessionEventType.USER_MESSAGE, UserMessageData(content="go", message_id=MSG_ID)),
            event(
                SessionEventType.ASSISTANT_MESSAGE_DELTA,
                AssistantMessageDeltaData(
                    delta_content="hello",
                    message_id="m1",
                ),
            ),
            event(
                SessionEventType.ASSISTANT_MESSAGE,
                AssistantMessageData(
                    content="hello",
                    message_id="m1",
                    originating_message_id=MSG_ID,
                    model=MODEL,
                ),
            ),
            event(
                SessionEventType.ASSISTANT_USAGE,
                AssistantUsageData(
                    model=MODEL,
                    input_tokens=10,
                    output_tokens=1,
                    cache_read_tokens=4,
                ),
            ),
            event(SessionEventType.SESSION_IDLE, SessionIdleData()),
        ]
    )
    assert out == [
        TextDelta(text="hello"),
        TokenUsage(model=MODEL, input_tokens=10, output_tokens=1, cache_read_tokens=4),
    ]


def test_full_response_correlates_without_user_message_event():
    _, out = _stream(
        [
            _ev(
                SessionEventType.ASSISTANT_MESSAGE,
                content="reply",
                message_id="m1",
                originating_message_id=MSG_ID,
                model=MODEL,
            ),
            _idle(),
        ]
    )
    assert out == [TextDelta(text="reply")]


def test_run_streamed_emits_token_usage():
    _, out = _stream(
        [
            _started(),
            _ev(
                SessionEventType.ASSISTANT_USAGE,
                model=MODEL,
                input_tokens=100,
                output_tokens=20,
                cache_read_tokens=64,
                cache_write_tokens=None,
            ),
            _idle(),
        ]
    )
    assert out == [
        TokenUsage(model=MODEL, input_tokens=100, output_tokens=20, cache_read_tokens=64, cache_write_tokens=0)
    ]


def test_run_streamed_ignores_sub_agent_output_but_counts_usage():
    _, out = _stream(
        [
            _started(),
            _ev(SessionEventType.ASSISTANT_MESSAGE_DELTA, "sub", delta_content="sub", message_id="s1"),
            _ev(SessionEventType.SESSION_IDLE, "sub", aborted=False),
            _ev(SessionEventType.ASSISTANT_USAGE, "sub", model="other-model", input_tokens=5, output_tokens=1),
            _delta("main"),
            _idle(),
        ]
    )
    assert out == [TokenUsage(model="other-model", input_tokens=5, output_tokens=1), TextDelta(text="main")]


def _tool_events(name: str = "echo", success: bool = True) -> list[_Event]:
    result = SimpleNamespace(contents=[SimpleNamespace(text="tool-out")], content="ignored")
    return [
        _ev(SessionEventType.TOOL_EXECUTION_START, tool_call_id="c1", tool_name=f"tb-{name}", mcp_tool_name=name),
        _ev(
            SessionEventType.TOOL_EXECUTION_COMPLETE,
            tool_call_id="c1",
            success=success,
            result=result if success else None,
            error=None if success else SimpleNamespace(message="denied", code="denied"),
        ),
    ]


def test_run_streamed_correlates_tool_name_from_start_event():
    handle, out = _stream([_started(), *_tool_events(), _delta("post-tool"), _idle()])
    assert out == [ToolEnd(tool_name="echo", text="tool-out"), TextDelta(text="post-tool")]
    assert handle.session.aborted == 0


def test_run_streamed_skips_failed_tool():
    _, out = _stream([_started(), *_tool_events(success=False), _idle()])
    assert out == []


def test_run_streamed_tool_falls_back_to_content_string():
    complete = _ev(
        SessionEventType.TOOL_EXECUTION_COMPLETE,
        tool_call_id="c9",
        success=True,
        result=SimpleNamespace(contents=None, content="plain"),
    )
    _, out = _stream([_started(), complete, _idle()])
    assert out == [ToolEnd(tool_name="", text="plain")]


def test_run_streamed_aborts_after_tool_when_excluded():
    handle, out = _stream([_started(), *_tool_events(), _delta("should-not-see"), _idle()], exclude=True)
    assert handle.session.aborted == 1
    assert out == [ToolEnd(tool_name="echo", text="tool-out")]


@pytest.mark.parametrize(
    ("data", "exc"),
    [
        ({"error_type": "rate_limit", "message": "slow down", "status_code": 429}, BackendRateLimitError),
        ({"error_type": "query", "message": "slow down", "status_code": 429}, BackendRateLimitError),
        ({"error_type": "authentication", "message": "bad creds", "status_code": 401}, BackendBadRequestError),
        ({"error_type": "query", "message": "no such model", "status_code": 404}, BackendBadRequestError),
        ({"error_type": "query", "message": "gateway", "status_code": 504}, BackendTimeoutError),
        ({"error_type": "quota", "message": "boom", "status_code": None}, BackendUnexpectedError),
    ],
)
def test_run_streamed_session_error_mapping(data, exc):
    handle = _handle([_started(), _ev(SessionEventType.SESSION_ERROR, originating_message_id=None, **data)])
    with pytest.raises(exc, match=data["message"]):
        asyncio.run(_collect(handle))
    assert handle.session.aborted == 1


def test_run_streamed_session_error_before_start_is_raised():
    handle = _handle(
        [
            _ev(
                SessionEventType.SESSION_ERROR,
                error_type="authentication",
                message="denied",
                status_code=None,
                originating_message_id=MSG_ID,
            )
        ]
    )
    with pytest.raises(BackendBadRequestError, match="denied"):
        asyncio.run(_collect(handle))


def test_policy_block_without_user_message_fails_instead_of_waiting():
    handle = _handle(
        [
            _ev(
                SessionEventType.SESSION_WARNING,
                warning_type="policy_blocked",
                message="Request blocked by policy",
                originating_message_id=MSG_ID,
            ),
            _idle(),
        ]
    )
    with pytest.raises(BackendBadRequestError, match="blocked by policy"):
        asyncio.run(_collect(handle))
    assert handle.session.aborted == 1


def test_run_streamed_ignores_error_for_other_message():
    _, out = _stream(
        [
            _started(),
            _ev(
                SessionEventType.SESSION_ERROR,
                error_type="query",
                message="x",
                status_code=None,
                originating_message_id="other",
            ),
            _delta("ok"),
            _idle(),
        ]
    )
    assert out == [TextDelta(text="ok")]


def test_run_streamed_unexpected_abort_raises():
    with pytest.raises(BackendUnexpectedError, match="aborted"):
        _stream([_started(), _delta("partial"), _idle(aborted=True)])


def test_autopilot_idle_does_not_complete_response():
    _, out = _stream(
        [
            _started(),
            _ev(SessionEventType.SESSION_IDLE, mode="autopilot"),
            _delta("final"),
            _idle(),
        ]
    )
    assert out == [TextDelta(text="final")]


@pytest.mark.parametrize(
    "event",
    [
        _ev(SessionEventType.SESSION_MODEL_CHANGE, new_model="gpt-4.1", previous_model=MODEL, cause="fallback"),
        _ev(SessionEventType.ASSISTANT_USAGE, model="gpt-4.1", input_tokens=1, output_tokens=1),
        _ev(SessionEventType.ASSISTANT_MESSAGE, content="hi", message_id="m1", model="gpt-4.1"),
    ],
    ids=lambda e: e.type.value,
)
def test_run_streamed_rejects_model_switch(event):
    handle = _handle([_started(), event, _idle()])
    with pytest.raises(BackendUnexpectedError, match="refusing"):
        asyncio.run(_collect(handle))
    assert handle.session.aborted == 1


def test_run_streamed_auto_model_accepts_routed_model(caplog):
    caplog.set_level("INFO")
    _, out = _stream(
        [
            _started(),
            _ev(SessionEventType.SESSION_MODEL_CHANGE, new_model="gpt-6-luna", previous_model="auto"),
            _delta("routed"),
            _ev(SessionEventType.ASSISTANT_USAGE, model="gpt-6-luna", input_tokens=1, output_tokens=1),
            _idle(),
        ],
        model="auto",
    )
    assert out == [TextDelta(text="routed"), TokenUsage(model="gpt-6-luna", input_tokens=1, output_tokens=1)]
    assert "'auto' routed to 'gpt-6-luna'" in caplog.text


def test_run_streamed_rejects_model_switch_in_stale_events():
    handle = _handle([], stale=[_ev(SessionEventType.SESSION_START, selected_model="gpt-4.1")])
    with pytest.raises(BackendUnexpectedError, match=r"gpt-4\.1"):
        asyncio.run(_collect(handle))
    assert handle.session.sent == []


@pytest.mark.parametrize(
    ("raised", "exc"),
    [(TimeoutError("t"), BackendTimeoutError), (RuntimeError("closed"), BackendUnexpectedError)],
)
def test_run_streamed_send_errors_are_translated(raised, exc):
    session = _FakeSession()
    session.send_error = raised
    with pytest.raises(exc) as info:
        _stream([], session=session)
    assert info.value.__cause__ is raised
    assert session.aborted == 1


def test_cancellation_during_send_aborts_session():
    session = _FakeSession()
    session.send_error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        _stream([], session=session)
    assert session.aborted == 1


def test_cancellation_while_waiting_for_events_aborts_session():
    async def _run():
        sent = asyncio.Event()

        class _WaitingSession(_FakeSession):
            async def send(self, _prompt):
                sent.set()
                return MSG_ID

        handle = _handle([], session=_WaitingSession())
        task = asyncio.create_task(_collect(handle))
        await sent.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert handle.session.aborted == 1

    asyncio.run(_run())


def test_run_streamed_aborts_session_on_consumer_break():
    handle = _handle([_started(), _delta("hello"), _delta("world"), _idle()])

    async def _run():
        async for _ev_ in CopilotSDKBackend().run_streamed(handle, "go", max_turns=10):
            break

    asyncio.run(_run())
    assert handle.session.aborted == 1


# --- build ------------------------------------------------------------------


class _FakeClient:
    instances: ClassVar[list[_FakeClient]] = []
    models: ClassVar[list[Any]] = []
    start_error: ClassVar[BaseException | None] = None
    create_error: ClassVar[BaseException | None] = None
    authenticated: ClassVar[bool] = True
    session_events: ClassVar[list[Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        # The real public constructor validates options without starting a runtime.
        _SDK_CLIENT(**kwargs)
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.create_kwargs: dict[str, Any] | None = None
        self.session: _FakeSession | None = None
        _FakeClient.instances.append(self)

    async def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    async def list_models(self) -> list[Any]:
        return self.models

    async def get_auth_status(self):
        return copilot.GetAuthStatusResponse(isAuthenticated=self.authenticated)

    async def create_session(self, **kwargs: Any) -> _FakeSession:
        _CREATE_SIGNATURE.bind(self, **kwargs)
        if self.create_error is not None:
            raise self.create_error
        self.create_kwargs = kwargs
        self.session = _FakeSession(self.session_events, kwargs["on_event"])
        return self.session

    async def stop(self) -> None:
        self.stopped = True


def _model(model_id: str, state: str | None = "enabled") -> Any:
    return SimpleNamespace(id=model_id, policy=SimpleNamespace(state=state) if state else None)


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setattr(_FakeClient, "instances", [])
    monkeypatch.setattr(_FakeClient, "models", [_model(MODEL), _model("off", "disabled")])
    monkeypatch.setattr(_FakeClient, "start_error", None)
    monkeypatch.setattr(_FakeClient, "create_error", None)
    monkeypatch.setattr(_FakeClient, "authenticated", True)
    monkeypatch.setattr(_FakeClient, "session_events", [])
    monkeypatch.setattr(copilot, "CopilotClient", _FakeClient)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", SECRET)
    return _FakeClient


def _build(**overrides):
    overrides.setdefault("model", MODEL)
    overrides.setdefault("token_env", "COPILOT_GITHUB_TOKEN")
    spec = _spec(**overrides)
    return asyncio.run(CopilotSDKBackend().build(spec))


def test_build_uses_explicit_token_and_pins_session(fake_client, caplog):
    caplog.set_level("DEBUG")
    native = SimpleNamespace(
        namespace="ns",
        list_tools_unfiltered=AsyncMock(
            return_value=[
                Tool(name="nsecho", inputSchema={"type": "object"}),
                Tool(name="nsbad_tool", inputSchema={"type": "object"}),
            ]
        ),
    )
    handle = _build(
        instructions="You are a security analyst.",
        blocked_tools=["bad_tool"],
        model_settings={"reasoning_effort": "max"},
        mcp_servers=[MCPServerSpec(name="tb", kind="stdio", params={"_native": native})],
    )
    (client,) = fake_client.instances
    assert client.kwargs == {"github_token": SECRET, "use_logged_in_user": False}
    assert client.started
    assert not client.stopped
    kw = client.create_kwargs
    assert kw["model"] == MODEL
    assert kw["system_message"] == {"mode": "replace", "content": "You are a security analyst."}
    assert kw["streaming"] is True
    assert kw["reasoning_effort"] == "max"
    assert kw["available_tools"].to_list() == ["custom:nsecho"]
    assert [tool.name for tool in kw["tools"]] == ["nsecho"]
    assert "mcp_servers" not in kw
    assert kw["enable_config_discovery"] is False
    assert kw["skip_custom_instructions"] is True
    assert kw["include_sub_agent_streaming_events"] is False
    assert kw["enable_skills"] is False
    assert kw["enable_file_hooks"] is False
    assert kw["enable_on_demand_instruction_discovery"] is False
    assert kw["memory"] == {"enabled": False}
    assert callable(kw["on_permission_request"])
    # No auto-mode handler: the SDK then declines any automatic model switch.
    assert "on_auto_mode_switch_request" not in kw
    assert handle.model == MODEL
    assert SECRET not in caplog.text


def test_build_strips_provider_prefix(fake_client):
    _build(model="openai/" + MODEL)
    assert fake_client.instances[0].create_kwargs["model"] == MODEL


def test_build_without_instructions_sends_no_system_message(fake_client):
    _build()
    assert fake_client.instances[0].create_kwargs["system_message"] is None


def test_build_missing_token_env_fails_closed(fake_client, monkeypatch):
    monkeypatch.delenv("SECLAB_TEST_MISSING_TOKEN", raising=False)
    spec = _spec(model=MODEL, token_env="SECLAB_TEST_MISSING_TOKEN")  # noqa: S106 - env var name
    with pytest.raises(BackendBadRequestError, match="SECLAB_TEST_MISSING_TOKEN"):
        asyncio.run(CopilotSDKBackend().build(spec))
    assert fake_client.instances == []


def test_build_defaults_to_copilot_github_token(fake_client):
    asyncio.run(CopilotSDKBackend().build(_spec(model=MODEL)))
    assert fake_client.instances[0].kwargs["github_token"] == SECRET


def test_build_explicit_token_overrides_default_identity(fake_client, monkeypatch):
    value = "different-offline-token"
    monkeypatch.setenv("SECLAB_TEST_TOKEN", value)
    _build(token_env="SECLAB_TEST_TOKEN")  # noqa: S106 - env var name
    assert fake_client.instances[0].kwargs["github_token"] == value
    assert fake_client.instances[0].kwargs["use_logged_in_user"] is False


def test_build_without_any_token_uses_sdk_default_auth(fake_client, monkeypatch):
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN")
    asyncio.run(CopilotSDKBackend().build(_spec(model=MODEL)))
    assert fake_client.instances[0].kwargs == {}


def test_build_authentication_failure_is_actionable_and_stops_client(fake_client):
    fake_client.authenticated = False
    with pytest.raises(BackendBadRequestError, match="authentication failed"):
        _build()
    (client,) = fake_client.instances
    assert client.create_kwargs is None
    assert client.stopped


@pytest.mark.parametrize(("model", "match"), [("nope", "not available"), ("off", "disabled")])
def test_build_rejects_unavailable_model_without_fallback(fake_client, model, match):
    with pytest.raises(BackendBadRequestError, match=match):
        _build(model=model)
    (client,) = fake_client.instances
    assert client.create_kwargs is None
    assert client.stopped


@pytest.mark.parametrize(
    ("attr", "raised", "exc"),
    [
        ("start_error", RuntimeError("auth failed"), BackendUnexpectedError),
        ("create_error", RuntimeError("bad mcp"), BackendUnexpectedError),
        ("create_error", TimeoutError(), BackendTimeoutError),
    ],
)
def test_build_failures_are_translated_and_stop_client(fake_client, monkeypatch, attr, raised, exc):
    monkeypatch.setattr(_FakeClient, attr, raised)
    with pytest.raises(exc) as info:
        _build()
    assert info.value.__cause__ is raised
    assert fake_client.instances[0].stopped
    assert SECRET not in str(info.value)


@pytest.mark.parametrize("attr", ["start_error", "create_error"])
def test_build_cancellation_stops_client(fake_client, monkeypatch, attr):
    monkeypatch.setattr(fake_client, attr, asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        _build()
    assert fake_client.instances[0].stopped


def test_build_and_stream_end_to_end(fake_client):
    fake_client.session_events = [
        _ev(SessionEventType.SESSION_START, selected_model=MODEL),
        _started(),
        _delta("CWE-476 is a NULL pointer dereference."),
        _ev(SessionEventType.ASSISTANT_USAGE, model=MODEL, input_tokens=10, output_tokens=8),
        _idle(),
    ]

    async def _run():
        backend = CopilotSDKBackend()
        handle = await backend.build(_spec(model=MODEL, token_env="COPILOT_GITHUB_TOKEN"))  # noqa: S106 - env var name
        try:
            return handle, [ev async for ev in backend.run_streamed(handle, "Explain", max_turns=1)]
        finally:
            await backend.aclose(handle)

    handle, out = asyncio.run(_run())
    assert out == [
        TextDelta(text="CWE-476 is a NULL pointer dereference."),
        TokenUsage(model=MODEL, input_tokens=10, output_tokens=8),
    ]
    assert handle.session.sent == ["Explain"]
    assert handle.session.disconnected == 1
    assert handle.client.stopped


def test_invalid_reasoning_effort_rejected():
    with pytest.raises(BackendBadRequestError, match="reasoning_effort"):
        _reasoning_effort({"reasoning_effort": "ludicrous"})


@pytest.mark.parametrize("model", ["", " ", "openai/", "openai/ "])
def test_normalize_model_rejects_empty(model):
    # Passing an empty model would let the SDK silently fall back to its
    # built-in default, which would invalidate any reproducibility
    # guarantee a taskflow makes about the model under test.
    with pytest.raises(BackendBadRequestError, match="model is required"):
        _normalize_model(model)


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


def test_aclose_stops_client_even_if_disconnect_is_cancelled():
    session = SimpleNamespace(disconnect=AsyncMock(side_effect=asyncio.CancelledError()))
    client = SimpleNamespace(stop=AsyncMock())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(CopilotSDKBackend().aclose(SimpleNamespace(client=client, session=session)))
    client.stop.assert_awaited_once()
