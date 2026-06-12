# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Anthropic SDK adapter."""

from __future__ import annotations

import pytest

from seclab_taskflow_agent.sdk import get_backend
from seclab_taskflow_agent.sdk.base import AgentSpec
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


# -- bearer_auth via provider registry --


def test_known_provider_uses_bearer_auth():
    """Known providers (CAPI, GitHub Models) should have bearer_auth=True."""
    from seclab_taskflow_agent.capi import get_provider

    provider = get_provider("https://api.githubcopilot.com")
    assert provider.bearer_auth is True

    provider = get_provider("https://models.github.ai/inference")
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
    """All Claude models support cache_control; default to on so callers
    get the cost savings without explicit opt-in. Explicit opt-out via
    prompt_caching=False remains available for proxies that don't support
    cache_control."""
    import asyncio

    from seclab_taskflow_agent.sdk.anthropic_sdk.backend import _AnthropicHandle

    captured = {}

    class _FakeStreamCtx:
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False
        def __aiter__(self):
            async def _gen():
                return
                yield
            return _gen()
        async def get_final_message(self):
            return type("M", (), {"stop_reason": "end_turn", "content": []})()

    class _FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return _FakeStreamCtx()

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    handle = _AnthropicHandle(
        client=_FakeClient(),
        system_prompt="",
        model="claude-mythos-5",
        max_tokens=100,
        tools=[],
        mcp_server_map={},
        model_settings={},
    )
    backend = AnthropicSDKBackend()

    async def _run():
        async for _ in backend.run_streamed(handle, "hi", max_turns=1):
            pass

    asyncio.run(_run())
    assert captured.get("cache_control") == {"type": "ephemeral"}, (
        f"expected default cache_control={{type: ephemeral}}, got {captured.get('cache_control')!r}"
    )


def test_prompt_caching_explicit_opt_out():
    """prompt_caching=False must suppress cache_control entirely (for
    callers pointed at proxies that don't support it)."""
    import asyncio

    from seclab_taskflow_agent.sdk.anthropic_sdk.backend import _AnthropicHandle

    captured = {}

    class _FakeStreamCtx:
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False
        def __aiter__(self):
            async def _gen():
                return
                yield
            return _gen()
        async def get_final_message(self):
            return type("M", (), {"stop_reason": "end_turn", "content": []})()

    class _FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return _FakeStreamCtx()

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    handle = _AnthropicHandle(
        client=_FakeClient(),
        system_prompt="",
        model="claude-mythos-5",
        max_tokens=100,
        tools=[],
        mcp_server_map={},
        model_settings={"prompt_caching": False},
    )
    backend = AnthropicSDKBackend()

    async def _run():
        async for _ in backend.run_streamed(handle, "hi", max_turns=1):
            pass

    asyncio.run(_run())
    assert "cache_control" not in captured, (
        f"cache_control should be absent when explicitly opted out, got {captured}"
    )


def test_prompt_caching_1h_ttl_passes_ttl_field():
    """When prompt_caching='1h', cache_control must include the 1h ttl."""
    import asyncio

    from seclab_taskflow_agent.sdk.anthropic_sdk.backend import _AnthropicHandle

    captured = {}

    class _FakeStreamCtx:
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False
        def __aiter__(self):
            async def _gen():
                return
                yield
            return _gen()
        async def get_final_message(self):
            return type("M", (), {"stop_reason": "end_turn", "content": []})()

    class _FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return _FakeStreamCtx()

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    handle = _AnthropicHandle(
        client=_FakeClient(),
        system_prompt="",
        model="claude-mythos-5",
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
    assert captured.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}, (
        f"expected cache_control with 1h ttl, got {captured.get('cache_control')!r}"
    )
