# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the openai-agents backend adapter."""

from __future__ import annotations

import asyncio

import pytest
from openai import APITimeoutError, BadRequestError, RateLimitError

from seclab_taskflow_agent import sdk
from seclab_taskflow_agent.sdk import errors
from seclab_taskflow_agent.sdk.base import AgentSpec, TextDelta
from seclab_taskflow_agent.sdk.openai_agents.backend import OpenAIAgentsBackend


@pytest.fixture
def backend() -> OpenAIAgentsBackend:
    return OpenAIAgentsBackend()


def _collect(async_iter):
    """Drain an async iterator into a list synchronously."""

    async def _run():
        return [event async for event in async_iter]

    return asyncio.run(_run())


def test_adapter_is_registered_on_import():
    backend = sdk.get_backend("openai_agents")
    assert backend.name == "openai_agents"


def test_validate_accepts_any_supported_spec(backend):
    backend.validate(
        AgentSpec(
            name="n",
            instructions="",
            model="m",
            model_settings={"temperature": 0.5, "parallel_tool_calls": True},
            in_handoff_graph=True,
            exclude_from_context=True,
        )
    )


class _FakeDelta:
    def __init__(self, text: str) -> None:
        self.delta = text


class _FakeRawTextEvent:
    # Duck-typed to match `event.type == "raw_response_event"` and
    # `isinstance(event.data, ResponseTextDeltaEvent)`. We bypass the
    # isinstance check by monkey-patching below in the corresponding
    # test.
    def __init__(self, text: str) -> None:
        self.type = "raw_response_event"
        self.data = _FakeDelta(text)


class _FakeOtherEvent:
    def __init__(self) -> None:
        self.type = "raw_response_event"
        self.data = object()  # not a ResponseTextDeltaEvent


class _FakeStreamedResult:
    def __init__(self, events):
        self._events = events
        self.cancel_called = False

    def stream_events(self):
        events = self._events

        async def iter_events():
            for ev in events:
                yield ev

        return iter_events()

    def cancel(self) -> None:
        self.cancel_called = True


class _FakeAgent:
    def __init__(self, events=None, raise_exc: Exception | None = None):
        self._events = events or []
        self._raise = raise_exc

    def run_streamed(self, prompt: str, *, max_turns: int):  # noqa: ARG002
        if self._raise is not None:
            raise self._raise
        return _FakeStreamedResult(self._events)


def test_run_streamed_emits_text_deltas(monkeypatch, backend):
    # Make isinstance(event.data, ResponseTextDeltaEvent) True for our fakes.
    from seclab_taskflow_agent.sdk.openai_agents import backend as backend_mod

    monkeypatch.setattr(backend_mod, "ResponseTextDeltaEvent", _FakeDelta)

    events = _collect(
        backend.run_streamed(
            _FakeAgent([_FakeRawTextEvent("hel"), _FakeRawTextEvent("lo")]),
            "prompt",
            max_turns=5,
        )
    )
    assert events == [TextDelta("hel"), TextDelta("lo")]


def test_run_streamed_ignores_non_text_raw_events(monkeypatch, backend):
    from seclab_taskflow_agent.sdk.openai_agents import backend as backend_mod

    monkeypatch.setattr(backend_mod, "ResponseTextDeltaEvent", _FakeDelta)

    events = _collect(
        backend.run_streamed(
            _FakeAgent([_FakeOtherEvent(), _FakeRawTextEvent("x")]),
            "p",
            max_turns=1,
        )
    )
    assert events == [TextDelta("x")]


def _make_rate_limit() -> RateLimitError:
    import httpx

    req = httpx.Request("POST", "https://example.test")
    resp = httpx.Response(429, request=req)
    return RateLimitError("rate", response=resp, body=None)


def _make_bad_request() -> BadRequestError:
    import httpx

    req = httpx.Request("POST", "https://example.test")
    resp = httpx.Response(400, request=req)
    return BadRequestError("bad", response=resp, body=None)


def _make_timeout() -> APITimeoutError:
    import httpx

    return APITimeoutError(request=httpx.Request("POST", "https://example.test"))


@pytest.mark.parametrize(
    ("raw_exc_factory", "neutral_cls"),
    [
        (_make_timeout, errors.BackendTimeoutError),
        (_make_rate_limit, errors.BackendRateLimitError),
        (_make_bad_request, errors.BackendBadRequestError),
    ],
)
def test_run_streamed_translates_openai_errors(backend, raw_exc_factory, neutral_cls):
    with pytest.raises(neutral_cls):
        _collect(backend.run_streamed(_FakeAgent(raise_exc=raw_exc_factory()), "p", max_turns=1))


def test_run_streamed_translates_max_turns():
    from agents.exceptions import MaxTurnsExceeded

    backend = OpenAIAgentsBackend()
    with pytest.raises(errors.BackendMaxTurnsError):
        _collect(
            backend.run_streamed(_FakeAgent(raise_exc=MaxTurnsExceeded("boom")), "p", max_turns=1)
        )


def test_run_streamed_translates_agents_exception():
    from agents.exceptions import AgentsException

    backend = OpenAIAgentsBackend()
    with pytest.raises(errors.BackendUnexpectedError):
        _collect(
            backend.run_streamed(_FakeAgent(raise_exc=AgentsException("weird")), "p", max_turns=1)
        )


def test_run_streamed_cancels_underlying_result(monkeypatch, backend):
    from seclab_taskflow_agent.sdk.openai_agents import backend as backend_mod

    monkeypatch.setattr(backend_mod, "ResponseTextDeltaEvent", _FakeDelta)
    agent = _FakeAgent([_FakeRawTextEvent("x")])
    _collect(backend.run_streamed(agent, "p", max_turns=1))
    # The adapter must call cancel() on the RunResultStreaming so the
    # background _run_impl_task does not leak after the stream drains.
    # We capture the most recently returned _FakeStreamedResult by
    # patching _FakeAgent.run_streamed via a wrapper.
    captured: list[_FakeStreamedResult] = []

    class _CapturingAgent(_FakeAgent):
        def run_streamed(self, prompt: str, *, max_turns: int):  # noqa: ARG002
            res = _FakeStreamedResult([_FakeRawTextEvent("y")])
            captured.append(res)
            return res

    _collect(backend.run_streamed(_CapturingAgent(), "p", max_turns=1))
    assert captured
    assert captured[0].cancel_called is True


def test_aclose_closes_taskagent_client(backend):
    closed: list[bool] = []

    class _FakeTaskAgent:
        async def close(self) -> None:
            closed.append(True)

    asyncio.run(backend.aclose(_FakeTaskAgent()))
    assert closed == [True]


def test_aclose_tolerates_objects_without_close(backend):
    asyncio.run(backend.aclose(object()))
    asyncio.run(backend.aclose(None))


def test_build_constructs_taskagent(monkeypatch, backend):
    """End-to-end build with TaskAgent mocked out."""
    captured: dict = {}

    class _FakeTaskAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.agent = object()

    from seclab_taskflow_agent import agent as agent_mod

    monkeypatch.setattr(agent_mod, "TaskAgent", _FakeTaskAgent)

    spec = AgentSpec(
        name="n",
        instructions="inst",
        model="gpt-5",
        model_settings={"temperature": 0.5},
        api_type="responses",
        endpoint="https://example.test",
    )
    result = asyncio.run(backend.build(spec))
    assert captured["name"] == "n"
    assert captured["model"] == "gpt-5"
    assert captured["api_type"] == "responses"
    assert captured["endpoint"] == "https://example.test"
    assert captured["handoffs"] == []
    assert captured["mcp_servers"] == []
    # No handoff graph participation → instructions passed through verbatim.
    assert captured["instructions"] == "inst"
    # model_settings is translated to ModelSettings(...) — just check it's non-None
    assert captured["model_settings"] is not None
    assert isinstance(result, _FakeTaskAgent)


def test_build_applies_handoff_prompt_when_flagged(monkeypatch, backend):
    """in_handoff_graph=True triggers prompt_with_handoff_instructions()."""
    captured: dict = {}

    class _FakeTaskAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.agent = object()

    from seclab_taskflow_agent import agent as agent_mod

    monkeypatch.setattr(agent_mod, "TaskAgent", _FakeTaskAgent)

    spec = AgentSpec(
        name="primary",
        instructions="BASE_INSTRUCTIONS",
        model="gpt-5",
        in_handoff_graph=True,
    )
    asyncio.run(backend.build(spec))
    # The wrapped instructions must contain the original text and must
    # be strictly longer (the openai-agents prefix adds handoff guidance).
    assert "BASE_INSTRUCTIONS" in captured["instructions"]
    assert len(captured["instructions"]) > len("BASE_INSTRUCTIONS")


def test_build_recurses_into_handoffs(monkeypatch, backend):
    """Handoff sub-specs are built via nested build() calls."""
    built: list[str] = []

    class _FakeTaskAgent:
        def __init__(self, **kwargs):
            built.append(kwargs["name"])
            self.agent = f"agent-{kwargs['name']}"

    from seclab_taskflow_agent import agent as agent_mod

    monkeypatch.setattr(agent_mod, "TaskAgent", _FakeTaskAgent)

    handoff = AgentSpec(name="sub", instructions="s", model="m", in_handoff_graph=True)
    primary = AgentSpec(
        name="main",
        instructions="p",
        model="m",
        handoffs=[handoff],
        in_handoff_graph=True,
    )
    asyncio.run(backend.build(primary))
    # Handoffs are constructed before the primary agent.
    assert built == ["sub", "main"]
