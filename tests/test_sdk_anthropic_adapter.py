# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Anthropic SDK adapter."""

from __future__ import annotations

from typing import Any

import pytest

from seclab_taskflow_agent.sdk import get_backend
from seclab_taskflow_agent.sdk.base import AgentSpec, TokenUsage
from seclab_taskflow_agent.sdk.anthropic_sdk.backend import (
    AnthropicSDKBackend,
    _mcp_tools_to_anthropic,
    _call_tool_result_to_text,
    _VALID_REASONING,
)
from seclab_taskflow_agent.sdk.errors import (
    BackendBadRequestError,
    BackendCapabilityError,
)


def _spec(**overrides) -> AgentSpec:
    base = {
        "name": "a",
        "instructions": "You are a test agent.",
        "model": "claude-opus-4.7",
    }
    base.update(overrides)
    return AgentSpec(**base)


def _make_fake_client(captured: dict, *, stop_reason: str = "end_turn", content: list | None = None, usage: Any = None):
    """Build a minimal fake Anthropic client that records messages.stream() kwargs.

    The returned client exposes ``client.messages.stream(**kwargs)``; ``kwargs`` is
    written into *captured* so tests can assert on what the backend would have sent
    to the real SDK.  The stream yields nothing and ``get_final_message()`` returns
    a stub with the requested ``stop_reason``/``content``/``usage``.
    """
    final_content = content if content is not None else []

    class _EmptyAsyncIter:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class _FakeStreamCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def __aiter__(self):
            return _EmptyAsyncIter()

        async def get_final_message(self):
            return type(
                "M",
                (),
                {"stop_reason": stop_reason, "content": final_content, "usage": usage},
            )()

    class _FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return _FakeStreamCtx()

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    return _FakeClient()


# -- Backend registration --


def test_get_backend_returns_anthropic_sdk_instance():
    backend = get_backend("anthropic_sdk")
    assert isinstance(backend, AnthropicSDKBackend)
    assert backend.name == "anthropic_sdk"


# -- validate() --


def test_validate_accepts_minimal_spec():
    AnthropicSDKBackend().validate(_spec())


def test_validate_rejects_handoffs():
    backend = AnthropicSDKBackend()
    with pytest.raises(BackendCapabilityError, match="handoffs"):
        backend.validate(_spec(handoffs=[_spec(name="b")]))


def test_validate_rejects_handoff_graph():
    backend = AnthropicSDKBackend()
    with pytest.raises(BackendCapabilityError, match="handoffs"):
        backend.validate(_spec(in_handoff_graph=True))


def test_validate_rejects_empty_model():
    backend = AnthropicSDKBackend()
    with pytest.raises(BackendBadRequestError, match="model is required"):
        backend.validate(_spec(model=""))


def test_validate_accepts_exclude_from_context():
    AnthropicSDKBackend().validate(_spec(exclude_from_context=True))


# -- _mcp_tools_to_anthropic() --


class _FakeTool:
    def __init__(self, name, description=None, input_schema=None):  # noqa: N803
        self.name = name
        self.description = description
        self.inputSchema = input_schema


def test_mcp_tools_to_anthropic_basic():
    tools = [
        _FakeTool(
            "read_file",
            "Read a file",
            {"type": "object", "properties": {"path": {"type": "string"}}},
        )
    ]
    result = _mcp_tools_to_anthropic(tools)
    assert len(result) == 1
    assert result[0]["name"] == "read_file"
    assert result[0]["description"] == "Read a file"
    assert result[0]["input_schema"]["properties"]["path"]["type"] == "string"


def test_mcp_tools_to_anthropic_none_description():
    """Tools with None description should fall back to tool name."""
    tools = [_FakeTool("my_tool", description=None)]
    result = _mcp_tools_to_anthropic(tools)
    assert result[0]["description"] == "my_tool"


def test_mcp_tools_to_anthropic_empty_description():
    """Tools with empty string description should fall back to tool name."""
    tools = [_FakeTool("my_tool", description="")]
    result = _mcp_tools_to_anthropic(tools)
    assert result[0]["description"] == "my_tool"


def test_mcp_tools_to_anthropic_no_schema():
    """Tools without inputSchema should get a default empty object schema."""
    tools = [_FakeTool("my_tool", "desc")]
    result = _mcp_tools_to_anthropic(tools)
    assert result[0]["input_schema"] == {"type": "object", "properties": {}}


def test_mcp_tools_to_anthropic_none_schema():
    """Tools with None inputSchema should get a default empty object schema."""
    tools = [_FakeTool("my_tool", "desc", input_schema=None)]
    result = _mcp_tools_to_anthropic(tools)
    assert result[0]["input_schema"] == {"type": "object", "properties": {}}


# -- _call_tool_result_to_text() --


class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeResult:
    def __init__(self, contents):
        self.content = contents


def test_call_tool_result_to_text_single():
    result = type("R", (), {"content": [_FakeContent("hello")]})()
    assert _call_tool_result_to_text(result) == "hello"


def test_call_tool_result_to_text_multiple():
    result = type("R", (), {"content": [_FakeContent("a"), _FakeContent("b")]})()
    assert _call_tool_result_to_text(result) == "a\nb"


def test_call_tool_result_to_text_empty():
    result = type("R", (), {"content": []})()
    text = _call_tool_result_to_text(result)
    assert isinstance(text, str)


def test_call_tool_result_to_text_preserves_empty_string():
    """A tool returning TextContent(text='') is reporting an explicit
    empty result. The helper must return '' verbatim, not fall back to
    str(result) (which is a noisy repr of the result object).

    Regression for the truthy-check bug: ``if text:`` was treating ''
    the same as None and dropping it, causing the empty content list
    branch to fire and emit ``str(result)`` to the model.
    """
    result = type("R", (), {"content": [_FakeContent("")]})()
    assert _call_tool_result_to_text(result) == ""


def test_call_tool_result_to_text_preserves_empty_among_nonempty():
    """Empty TextContent should join with neighbors as ''."""
    result = type("R", (), {"content": [_FakeContent("a"), _FakeContent(""), _FakeContent("b")]})()
    assert _call_tool_result_to_text(result) == "a\n\nb"


# -- bearer_auth via provider registry --


def test_known_provider_uses_bearer_auth():
    """Known providers like CAPI should have bearer_auth=True."""
    from seclab_taskflow_agent.capi import get_provider

    provider = get_provider("https://api.githubcopilot.com")
    assert provider.bearer_auth is True


def test_unknown_endpoint_uses_native_auth():
    """Unknown endpoints should default to native SDK auth (bearer_auth=False)."""
    from seclab_taskflow_agent.capi import get_provider

    provider = get_provider("https://api.anthropic.com")
    assert provider.bearer_auth is False
    assert provider.name == "custom"


def test_awf_proxy_inherits_upstream_bearer_auth(monkeypatch):
    """AWF proxy should inherit bearer_auth from the upstream provider."""
    from seclab_taskflow_agent.capi import get_provider

    monkeypatch.setenv("AWF_COPILOT_PROXY", "api.githubcopilot.com")
    provider = get_provider("http://localhost:8080")
    assert provider.bearer_auth is True
    assert provider.base_url == "http://localhost:8080/"


# -- reasoning validation --


def test_valid_reasoning_values():
    assert _VALID_REASONING == ("low", "medium", "high", "max")


# -- reasoning effort validation (runtime) --


def test_invalid_reasoning_effort_not_in_valid():
    """Invalid reasoning.effort values should not be in _VALID_REASONING."""
    from seclab_taskflow_agent.sdk.anthropic_sdk.backend import _VALID_REASONING

    assert "ultra" not in _VALID_REASONING
    assert "high" in _VALID_REASONING
    assert "low" in _VALID_REASONING
    assert "max" in _VALID_REASONING


def test_invalid_reasoning_effort_raises_at_runtime():
    """run_streamed raises BackendBadRequestError for invalid effort."""
    import asyncio

    from seclab_taskflow_agent.sdk.anthropic_sdk.backend import _AnthropicHandle

    handle = _AnthropicHandle(
        client=None,
        system_prompt="",
        model="test",
        max_tokens=100,
        tools=[],
        mcp_server_map={},
        model_settings={"reasoning": {"effort": "ultra"}},
    )
    backend = AnthropicSDKBackend()

    async def _run():
        async for _ in backend.run_streamed(handle, "hi", max_turns=1):
            pass

    with pytest.raises(BackendBadRequestError, match="invalid reasoning effort"):
        asyncio.run(_run())


# -- prompt caching --


def test_prompt_caching_enabled_by_default():
    """Caching defaults on: the stable prefix (system prompt + last tool
    definition) carries a block-level cache_control breakpoint, and no
    top-level cache_control param is sent -- CAPI's native /v1/messages
    endpoints reject the top-level form but accept the block-level one."""
    import asyncio

    from seclab_taskflow_agent.sdk.anthropic_sdk.backend import _AnthropicHandle

    captured: dict = {}
    tools = [
        {"name": "a", "description": "", "input_schema": {"type": "object"}},
        {"name": "b", "description": "", "input_schema": {"type": "object"}},
    ]
    handle = _AnthropicHandle(
        client=_make_fake_client(captured),
        system_prompt="You are a helpful auditor.",
        model="claude-opus-4.7",
        max_tokens=100,
        tools=tools,
        mcp_server_map={},
        model_settings={},
    )
    backend = AnthropicSDKBackend()

    async def _run():
        async for _ in backend.run_streamed(handle, "hi", max_turns=1):
            pass

    asyncio.run(_run())
    # No top-level cache_control param.
    assert "cache_control" not in captured, (
        f"top-level cache_control must not be sent, got {captured.get('cache_control')!r}"
    )
    # System prompt is sent as a content block carrying the breakpoint.
    assert captured["system"] == [
        {
            "type": "text",
            "text": "You are a helpful auditor.",
            "cache_control": {"type": "ephemeral"},
        }
    ], f"expected block-level system cache_control, got {captured.get('system')!r}"
    # The breakpoint lands on the last tool; earlier tools are untouched.
    assert captured["tools"][-1].get("cache_control") == {"type": "ephemeral"}
    assert "cache_control" not in captured["tools"][0]


def test_prompt_caching_explicit_opt_out():
    """prompt_caching=False must suppress all cache_control: the system prompt
    stays a plain string and tool definitions carry no breakpoint."""
    import asyncio

    from seclab_taskflow_agent.sdk.anthropic_sdk.backend import _AnthropicHandle

    captured: dict = {}
    handle = _AnthropicHandle(
        client=_make_fake_client(captured),
        system_prompt="You are a helpful auditor.",
        model="claude-opus-4.7",
        max_tokens=100,
        tools=[{"name": "a", "description": "", "input_schema": {"type": "object"}}],
        mcp_server_map={},
        model_settings={"prompt_caching": False},
    )
    backend = AnthropicSDKBackend()

    async def _run():
        async for _ in backend.run_streamed(handle, "hi", max_turns=1):
            pass

    asyncio.run(_run())
    assert "cache_control" not in captured
    assert captured["system"] == "You are a helpful auditor.", (
        f"system should stay a plain string when caching is off, got {captured.get('system')!r}"
    )
    assert "cache_control" not in captured["tools"][0]


def test_prompt_caching_1h_ttl_passes_ttl_field():
    """prompt_caching='1h' sets the extended ttl on the block-level breakpoint."""
    import asyncio

    from seclab_taskflow_agent.sdk.anthropic_sdk.backend import _AnthropicHandle

    captured: dict = {}
    handle = _AnthropicHandle(
        client=_make_fake_client(captured),
        system_prompt="You are a helpful auditor.",
        model="claude-opus-4.7",
        max_tokens=100,
        tools=[],
        mcp_server_map={},
        model_settings={"prompt_caching": "1h"},
    )
    backend = AnthropicSDKBackend()

    async def _run():
        async for _ in backend.run_streamed(handle, "hi", max_turns=1):
            pass

    asyncio.run(_run())
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}, (
        f"expected system block cache_control with 1h ttl, got {captured.get('system')!r}"
    )


def test_run_streamed_emits_token_usage():
    """The response usage (including prompt-cache tokens) is surfaced as a
    neutral TokenUsage stream event."""
    import asyncio

    from seclab_taskflow_agent.sdk.anthropic_sdk.backend import _AnthropicHandle

    usage = type(
        "U",
        (),
        {
            "input_tokens": 9,
            "output_tokens": 20,
            "cache_read_input_tokens": 1321,
            "cache_creation_input_tokens": 0,
        },
    )()
    captured: dict = {}
    handle = _AnthropicHandle(
        client=_make_fake_client(captured, usage=usage),
        system_prompt="You are a helpful auditor.",
        model="claude-opus-4.7",
        max_tokens=100,
        tools=[],
        mcp_server_map={},
        model_settings={},
    )
    backend = AnthropicSDKBackend()

    async def _run():
        return [ev async for ev in backend.run_streamed(handle, "hi", max_turns=1)]

    events = asyncio.run(_run())
    assert TokenUsage(
        model="claude-opus-4.7",
        input_tokens=9,
        output_tokens=20,
        cache_read_tokens=1321,
        cache_write_tokens=0,
    ) in events


# -- blocked_tools filtering --


def test_blocked_tools_matches_raw_name_against_namespaced_tool(monkeypatch):
    """Regression: taskflow YAML blocked_tools uses raw (un-namespaced)
    names like 'read_file', but list_tools_unfiltered() returns
    namespace-prefixed names like '{hash}read_file'. The filter must
    match the raw name against the un-prefixed portion of the
    namespaced tool, otherwise blocking is silently bypassed.

    See PR #265 review thread and openai_agents/copilot_sdk for
    how blocked_tools are consumed elsewhere (both use raw names).
    """
    monkeypatch.setenv("AI_API_TOKEN", "test-token")
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from seclab_taskflow_agent.mcp_utils import MCPNamespaceWrap, compress_name
    from seclab_taskflow_agent.sdk.base import MCPServerSpec

    class _FakeTool:
        def __init__(self, name):
            self.name = name
            self.description = ""
            self.inputSchema = {}

        def copy(self):
            t = _FakeTool(self.name)
            return t

    # Build a wrapper whose session.list_tools returns two raw tools.
    # list_tools_unfiltered() will return them with namespace prefix.
    obj = MagicMock()
    obj.name = "RepoContext"
    ns = compress_name("RepoContext")
    obj.session = MagicMock()
    obj.session.list_tools = AsyncMock(
        return_value=type("R", (), {"tools": [_FakeTool("read_file"), _FakeTool("safe_helper")]})()
    )
    wrap = MCPNamespaceWrap(confirms=[], obj=obj)

    spec = AgentSpec(
        name="t",
        instructions="",
        model="claude-sonnet-4.5",
        mcp_servers=[MCPServerSpec(name="rc", kind="stdio", params={"_native": wrap})],
        blocked_tools=["read_file"],  # raw name from YAML
    )
    backend = AnthropicSDKBackend()
    handle = asyncio.run(backend.build(spec))

    # The blocked tool must be absent from both the tool list AND the
    # server map keys (which use the namespaced form).
    tool_names = [t["name"] for t in handle.tools]
    assert f"{ns}read_file" not in tool_names, (
        f"blocked raw name 'read_file' should have filtered out '{ns}read_file'; "
        f"got tools: {tool_names}"
    )
    assert f"{ns}safe_helper" in tool_names, (
        f"non-blocked tool 'safe_helper' should still be present; got: {tool_names}"
    )
    assert f"{ns}read_file" not in handle.mcp_server_map
    assert f"{ns}safe_helper" in handle.mcp_server_map


def test_blocked_tools_also_matches_already_namespaced_name(monkeypatch):
    """Backwards-compat: if a caller already passes the namespaced name
    in blocked_tools (e.g. they computed it externally), it should still
    match. The filter checks both forms."""
    monkeypatch.setenv("AI_API_TOKEN", "test-token")
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from seclab_taskflow_agent.mcp_utils import MCPNamespaceWrap, compress_name
    from seclab_taskflow_agent.sdk.base import MCPServerSpec

    class _FakeTool:
        def __init__(self, name):
            self.name = name
            self.description = ""
            self.inputSchema = {}

        def copy(self):
            return _FakeTool(self.name)

    obj = MagicMock()
    obj.name = "RepoContext"
    ns = compress_name("RepoContext")
    obj.session = MagicMock()
    obj.session.list_tools = AsyncMock(
        return_value=type("R", (), {"tools": [_FakeTool("read_file")]})()
    )
    wrap = MCPNamespaceWrap(confirms=[], obj=obj)

    spec = AgentSpec(
        name="t",
        instructions="",
        model="claude-sonnet-4.5",
        mcp_servers=[MCPServerSpec(name="rc", kind="stdio", params={"_native": wrap})],
        blocked_tools=[f"{ns}read_file"],  # already namespaced
    )
    backend = AnthropicSDKBackend()
    handle = asyncio.run(backend.build(spec))

    assert handle.tools == [], (
        f"blocked namespaced name should filter out the tool; got: {handle.tools}"
    )


# -- token validation --


def test_build_raises_bad_request_when_no_token_available(monkeypatch):
    """build() must fail loudly when no API token can be resolved.

    Otherwise the Anthropic client gets created with an empty 'Bearer '
    header and the failure surfaces later as an opaque 401 mid-stream
    instead of a clear BackendBadRequestError at build time.

    Clears every variable consulted by ``capi.get_AI_token``
    (``AI_API_TOKEN`` then ``COPILOT_TOKEN``) to keep the test
    deterministic regardless of the runner's ambient environment.
    """
    import asyncio

    # Must clear *every* env var the token chain consults; missing
    # COPILOT_TOKEN here would make the test flaky on runners that
    # happen to have it set (e.g. CI machines authed to copilot).
    monkeypatch.delenv("AI_API_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_TOKEN", raising=False)

    spec = AgentSpec(
        name="t",
        instructions="",
        model="claude-sonnet-4.5",
        endpoint="https://api.githubcopilot.com",
    )
    backend = AnthropicSDKBackend()
    with pytest.raises(BackendBadRequestError, match="no API token"):
        asyncio.run(backend.build(spec))


# -- exception mapping (4xx -> BackendBadRequestError) --


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 409, 422])
def test_4xx_api_status_errors_map_to_bad_request(monkeypatch, status_code):
    """Any 4xx APIStatusError must surface as BackendBadRequestError so the
    runner logs it as a request error rather than an internal exception.
    Previously only BadRequestError (400) was mapped, leaving auth/permission/
    not-found errors (401/403/404) to surface as BackendUnexpectedError."""
    import asyncio
    import anthropic
    import httpx

    from seclab_taskflow_agent.sdk.anthropic_sdk.backend import _AnthropicHandle

    response = httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://test.example/v1/messages"),
    )

    class _RaisingStreamCtx:
        async def __aenter__(self):
            raise anthropic.APIStatusError(
                f"http {status_code}", response=response, body=None
            )

        async def __aexit__(self, *exc):
            return False

    class _FakeMessages:
        def stream(self, **kwargs):  # noqa: ARG002
            return _RaisingStreamCtx()

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    handle = _AnthropicHandle(
        client=_FakeClient(),
        system_prompt="",
        model="claude-opus-4.7",
        max_tokens=100,
        tools=[],
        mcp_server_map={},
        model_settings={"prompt_caching": False},
    )
    backend = AnthropicSDKBackend()

    async def _run():
        async for _ in backend.run_streamed(handle, "hi", max_turns=1):
            pass

    with pytest.raises(BackendBadRequestError):
        asyncio.run(_run())


def test_5xx_api_status_errors_map_to_unexpected(monkeypatch):
    """5xx APIStatusError must still surface as BackendUnexpectedError (not
    BackendBadRequestError); the request itself was well-formed."""
    import asyncio
    import anthropic
    import httpx

    from seclab_taskflow_agent.sdk.anthropic_sdk.backend import _AnthropicHandle
    from seclab_taskflow_agent.sdk.errors import BackendUnexpectedError

    response = httpx.Response(
        status_code=503,
        request=httpx.Request("POST", "https://test.example/v1/messages"),
    )

    class _RaisingStreamCtx:
        async def __aenter__(self):
            raise anthropic.InternalServerError(
                "service unavailable", response=response, body=None
            )

        async def __aexit__(self, *exc):
            return False

    class _FakeMessages:
        def stream(self, **kwargs):  # noqa: ARG002
            return _RaisingStreamCtx()

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    handle = _AnthropicHandle(
        client=_FakeClient(),
        system_prompt="",
        model="claude-opus-4.7",
        max_tokens=100,
        tools=[],
        mcp_server_map={},
        model_settings={"prompt_caching": False},
    )
    backend = AnthropicSDKBackend()

    async def _run():
        async for _ in backend.run_streamed(handle, "hi", max_turns=1):
            pass

    with pytest.raises(BackendUnexpectedError):
        asyncio.run(_run())
