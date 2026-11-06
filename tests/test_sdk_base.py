# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the backend factory and base types."""

from __future__ import annotations

import inspect

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


def test_resolve_backend_copilot_endpoint_does_not_auto_select(monkeypatch):
    """Backend selection is always explicit -- endpoint URL is not used."""
    monkeypatch.delenv("SECLAB_TASKFLOW_BACKEND", raising=False)
    assert (
        sdk.resolve_backend_name(endpoint="https://api.githubcopilot.com")
        == "openai_agents"
    )


def test_resolve_backend_explicit_overrides_endpoint(monkeypatch):
    monkeypatch.delenv("SECLAB_TASKFLOW_BACKEND", raising=False)
    assert (
        sdk.resolve_backend_name(
            explicit="anthropic_sdk",
            endpoint="https://api.githubcopilot.com",
        )
        == "anthropic_sdk"
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


def test_every_backend_adapter_translates_transport_failures():
    """A new adapter must not be able to forget this quietly.

    Transport exceptions are in neither the vendor SDKs' hierarchies nor
    the builtins', so an adapter that does not translate them lets them
    escape past the runner's retry entirely. That is what left a dropped
    response stream unretried, and it was missed in copilot_sdk on the
    first pass precisely because nothing checked.

    Asserted across the registry rather than per adapter, so adding a
    backend without translating fails here rather than in production.

    Necessary but not sufficient, and copilot_sdk is the proof: it had the
    handler while the one call that actually does inline network I/O sat
    outside the try, so the clause could never fire. A new adapter still
    needs a behavioural test that drives a transport failure through it.
    """
    import importlib
    import re

    from seclab_taskflow_agent.sdk import _KNOWN

    for name in _KNOWN:
        mod = importlib.import_module(
            f"seclab_taskflow_agent.sdk.{name}.backend"
        )
        src = inspect.getsource(mod)
        # Match the handler, not the import. An earlier version of this
        # test looked for the bare names and passed with the except clause
        # deleted, because the import line alone satisfied it.
        assert re.search(r"except\s+TRANSIENT_TRANSPORT_ERRORS\b", src), (
            f"{name} has no 'except TRANSIENT_TRANSPORT_ERRORS' handler; a "
            f"dropped stream would escape the runner's retry entirely. "
            f"See sdk/_transport.py"
        )
        assert re.search(r"raise\s+BackendTransportError\b", src), (
            f"{name} catches transport failures but does not raise the "
            f"neutral BackendTransportError the runner retries on"
        )
