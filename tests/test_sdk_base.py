# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the backend factory and base types."""

from __future__ import annotations

import pytest

from seclab_taskflow_agent import sdk
from seclab_taskflow_agent.sdk.base import AgentSpec, MCPServerSpec, TextDelta


def test_get_backend_returns_openai_agents_by_default():
    backend = sdk.get_backend("openai_agents")
    assert backend.name == "openai_agents"
    # Cached singleton.
    assert sdk.get_backend("openai_agents") is backend


def test_get_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown backend"):
        sdk.get_backend("nope")


def test_resolve_backend_explicit_wins(monkeypatch):
    monkeypatch.setenv("SECLAB_TASKFLOW_BACKEND", "copilot_sdk")
    assert sdk.resolve_backend_name(explicit="openai_agents") == "openai_agents"


def test_resolve_backend_env_var(monkeypatch):
    monkeypatch.setenv("SECLAB_TASKFLOW_BACKEND", "copilot_sdk")
    assert sdk.resolve_backend_name() == "copilot_sdk"


def test_resolve_backend_default_is_openai_agents(monkeypatch):
    monkeypatch.delenv("SECLAB_TASKFLOW_BACKEND", raising=False)
    assert sdk.resolve_backend_name() == "openai_agents"


def test_resolve_backend_copilot_endpoint_prefers_copilot_when_installed(monkeypatch):
    monkeypatch.delenv("SECLAB_TASKFLOW_BACKEND", raising=False)
    pytest.importorskip("copilot")
    assert (
        sdk.resolve_backend_name(endpoint="https://api.githubcopilot.com")
        == "copilot_sdk"
    )


def test_resolve_backend_copilot_endpoint_falls_back_when_missing(monkeypatch):
    monkeypatch.delenv("SECLAB_TASKFLOW_BACKEND", raising=False)
    # Force the optional import to fail by stashing a sentinel in sys.modules.
    import sys

    monkeypatch.setitem(sys.modules, "copilot", None)
    assert (
        sdk.resolve_backend_name(endpoint="https://api.githubcopilot.com")
        == "openai_agents"
    )


def test_resolve_backend_rejects_unknown(monkeypatch):
    monkeypatch.setenv("SECLAB_TASKFLOW_BACKEND", "nope")
    with pytest.raises(ValueError, match="Unknown backend"):
        sdk.resolve_backend_name()


def test_agent_spec_defaults_are_safe():
    spec = AgentSpec(name="n", instructions="", model="gpt-5")
    assert spec.model_settings == {}
    assert spec.mcp_servers == []
    assert spec.handoffs == []
    assert spec.blocked_tools == []
    assert spec.headless is False
    assert spec.in_handoff_graph is False


def test_text_delta_is_a_stream_event():
    event: sdk.StreamEvent = TextDelta(text="hi")
    assert event.text == "hi"


def test_mcp_server_spec_round_trip():
    spec = MCPServerSpec(name="x", kind="stdio", params={"command": "/bin/x"})
    assert spec.kind == "stdio"
    assert spec.params["command"] == "/bin/x"
