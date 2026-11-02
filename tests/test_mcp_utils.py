# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for MCPNamespaceWrap."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from agents.mcp import MCPServerStdio

from seclab_taskflow_agent.mcp_transport import ReconnectingMCPServerStdio
from seclab_taskflow_agent.mcp_utils import (
    MCPNamespaceWrap,
    _env_names,
    compress_name,
    mcp_client_params,
)


class _FakeTool:
    """Tool with a copy() method (mimics mcp.types.Tool)."""

    def __init__(self, name: str, description: str = "", input_schema: dict | None = None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {}

    def copy(self) -> _FakeTool:
        return _FakeTool(self.name, self.description, dict(self.inputSchema))


def _make_wrapper(server_name: str, session=None) -> MCPNamespaceWrap:
    """Construct an MCPNamespaceWrap around a mock underlying server."""
    obj = MagicMock()
    obj.name = server_name
    obj.session = session
    return MCPNamespaceWrap(confirms=[], obj=obj)


# -- list_tools_unfiltered() --


def test_list_tools_unfiltered_prefixes_names_from_session():
    """Tools from session.list_tools() should be namespace-prefixed."""
    tools = [_FakeTool("read_file", "Read a file"), _FakeTool("write_file", "Write a file")]
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=tools))
    wrapper = _make_wrapper("RepoContext", session=session)

    result = asyncio.run(wrapper.list_tools_unfiltered())

    ns = compress_name("RepoContext")
    assert len(result) == 2
    assert result[0].name == f"{ns}read_file"
    assert result[1].name == f"{ns}write_file"


def test_list_tools_unfiltered_follows_cursors_through_empty_pages():
    session = SimpleNamespace(
        list_tools=AsyncMock(
            side_effect=[
                SimpleNamespace(tools=[_FakeTool("first")], nextCursor="page2"),
                SimpleNamespace(tools=[], nextCursor="page3"),
                SimpleNamespace(tools=[_FakeTool("last")], nextCursor=None),
            ]
        )
    )
    wrapper = _make_wrapper("RepoContext", session=session)

    result = asyncio.run(wrapper.list_tools_unfiltered())

    assert [tool.name for tool in result] == [wrapper.namespace + "first", wrapper.namespace + "last"]
    assert session.list_tools.await_args_list == [call(), call(cursor="page2"), call(cursor="page3")]


def test_list_tools_unfiltered_rejects_repeated_cursors():
    page = SimpleNamespace(tools=[], nextCursor="repeated")
    session = SimpleNamespace(list_tools=AsyncMock(side_effect=[page, page]))
    wrapper = _make_wrapper("RepoContext", session=session)

    with pytest.raises(RuntimeError, match=r"repeated.*cursor"):
        asyncio.run(wrapper.list_tools_unfiltered())
    assert session.list_tools.await_count == 2


def test_list_tools_unfiltered_no_double_prefix_when_called_twice():
    """Calling list_tools_unfiltered twice should not double-prefix names."""
    session = MagicMock()
    # Fresh tools each call (mimics MCP session returning fresh objects)
    session.list_tools = AsyncMock(
        side_effect=lambda: SimpleNamespace(tools=[_FakeTool("get_repo")])
    )
    wrapper = _make_wrapper("RepoContext", session=session)

    async def _run():
        a = await wrapper.list_tools_unfiltered()
        b = await wrapper.list_tools_unfiltered()
        return a, b

    result1, result2 = asyncio.run(_run())

    ns = compress_name("RepoContext")
    assert result1[0].name == f"{ns}get_repo"
    assert result2[0].name == f"{ns}get_repo"
    # Crucially, the second result is NOT double-prefixed
    assert not result2[0].name.startswith(f"{ns}{ns}")


def test_list_tools_unfiltered_preserves_tool_attributes():
    """The copy of each tool should preserve description and input schema."""
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    tools = [_FakeTool("read_file", "Read a file", schema)]
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=tools))
    wrapper = _make_wrapper("RepoContext", session=session)

    result = asyncio.run(wrapper.list_tools_unfiltered())

    assert result[0].description == "Read a file"
    assert result[0].inputSchema == schema


def test_list_tools_unfiltered_raises_when_session_missing():
    """Should raise RuntimeError if the underlying server has no session yet."""
    wrapper = _make_wrapper("RepoContext", session=None)

    with pytest.raises(RuntimeError, match=r"no.*active MCP session"):
        asyncio.run(wrapper.list_tools_unfiltered())


@pytest.mark.parametrize("outcome", ["success", "listing-error", "cancelled", "connect-error"])
def test_unfiltered_reconnecting_listing_owns_its_session(monkeypatch, outcome):
    session = SimpleNamespace(
        list_tools=AsyncMock(
            side_effect=[
                SimpleNamespace(tools=[_FakeTool("first")], nextCursor="next"),
                SimpleNamespace(tools=[_FakeTool("last")], nextCursor=None),
            ]
        )
    )
    if outcome in ("listing-error", "cancelled"):
        error = asyncio.CancelledError() if outcome == "cancelled" else RuntimeError("listing failed")
        session.list_tools.side_effect = [
            SimpleNamespace(tools=[_FakeTool("first")], nextCursor="next"),
            error,
        ]
    events = []

    async def connect(server):
        events.append("connect")
        if outcome == "connect-error":
            raise RuntimeError("connection failed")
        server.session = session

    async def cleanup(server):
        events.append("cleanup")
        server.session = None

    monkeypatch.setattr(MCPServerStdio, "connect", connect)
    monkeypatch.setattr(MCPServerStdio, "cleanup", cleanup)
    transport = ReconnectingMCPServerStdio(params={"command": "unused"}, name="Repo")
    transport.list_tools = AsyncMock(side_effect=AssertionError("filtered listing must not be used"))
    wrapper = MCPNamespaceWrap([], transport)

    if outcome == "success":
        tools = asyncio.run(wrapper.list_tools_unfiltered())
        assert [tool.name for tool in tools] == [wrapper.namespace + "first", wrapper.namespace + "last"]
    else:
        exception = asyncio.CancelledError if outcome == "cancelled" else RuntimeError
        with pytest.raises(exception):
            asyncio.run(wrapper.list_tools_unfiltered())

    assert events == ["connect", "cleanup"]
    assert transport.session is None
    assert not transport.reconnecting_lock.locked()
    transport.list_tools.assert_not_awaited()
    expected_calls = [] if outcome == "connect-error" else [call(), call(cursor="next")]
    assert session.list_tools.await_args_list == expected_calls


def test_list_tools_unfiltered_does_not_share_state_with_caller():
    """Mutating returned tool names must not affect the underlying tools."""
    original = _FakeTool("read_file")
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[original]))
    wrapper = _make_wrapper("Repo", session=session)

    result = asyncio.run(wrapper.list_tools_unfiltered())
    result[0].name = "MUTATED"

    # Original tool should still have its name (copy() worked)
    assert original.name == "read_file"


def test_list_tools_unfiltered_idempotent_on_prefixed_input():
    """If the session returns a tool whose name is already namespace-prefixed
    (e.g. because of a cached/reused tool object), the prefix must NOT be
    applied a second time. Required for safe repeated/reentrant calls."""
    ns = compress_name("RepoContext")
    pre_prefixed = _FakeTool(f"{ns}read_file", "Read a file")
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[pre_prefixed]))
    wrapper = _make_wrapper("RepoContext", session=session)

    result = asyncio.run(wrapper.list_tools_unfiltered())

    # Result must have exactly one prefix, not two
    assert result[0].name == f"{ns}read_file"
    assert not result[0].name.startswith(f"{ns}{ns}")


# -- compress_name() --


@pytest.mark.parametrize(
    "server_name",
    # "ContainerShell" is one whose sha256 starts with a digit, which is what
    # made this show up as an intermittent failure of unrelated toolboxes.
    ["ContainerShell", "RepoContext", "FindingLedger", "RepoSurvey", "", "x" * 200],
)
def test_compress_name_starts_with_a_letter(server_name):
    """Gemini rejects a function name that starts with a digit.

    A namespace is prefixed to every tool a server exposes, so a digest
    beginning with a digit produces names like `8fb2adfa03efcontainer_shell_exec`.
    Gemini answers 400 `invalid_request_body` to the whole request, taking down
    every tool in the run and not just the offending server's.
    """
    assert compress_name(server_name)[0].isalpha()


def test_compress_name_is_stable_and_distinct():
    assert compress_name("RepoContext") == compress_name("RepoContext")
    assert compress_name("RepoContext") != compress_name("RepoSurvey")


def test_compress_name_leaves_room_under_the_64_character_limit():
    """The prefix must not eat the headroom the compression exists to create."""
    assert len(compress_name("x" * 200)) + len("a_very_long_tool_name_indeed") <= 64


# -- list_tools() (regression) --


def test_list_tools_existing_behaviour_unchanged():
    """Existing list_tools() should still forward args and prefix names."""
    tools = [_FakeTool("read_file")]
    obj = MagicMock()
    obj.name = "RepoContext"
    obj.list_tools = AsyncMock(return_value=tools)
    obj.session = MagicMock()
    wrapper = MCPNamespaceWrap(confirms=[], obj=obj)

    result = asyncio.run(wrapper.list_tools(run_context="ctx", agent="agent"))

    obj.list_tools.assert_awaited_once_with(run_context="ctx", agent="agent")
    ns = compress_name("RepoContext")
    assert result[0].name == f"{ns}read_file"


def test_call_tool_reports_confirmation_refusal_as_an_error(monkeypatch):
    transport = SimpleNamespace(name="Repo", call_tool=AsyncMock())
    wrapper = MCPNamespaceWrap(["write_file"], transport)
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")

    result = asyncio.run(wrapper.call_tool(wrapper.namespace + "write_file", {"path": "example.txt"}))

    assert result.isError is True
    assert result.content[0].text == "Tool call not allowed."
    transport.call_tool.assert_not_awaited()


# -- secret redaction in debug logs --


def test_mcp_client_params_logs_env_names_not_secret_values(monkeypatch, caplog):
    # A tool-call environment routinely resolves credentials like GH_TOKEN;
    # debug logging must record only the variable names, never the values.
    sentinel = "do-not-log-this-value"
    monkeypatch.setenv("GH_TOKEN", sentinel)
    server_params = SimpleNamespace(
        kind="stdio",
        reconnecting=False,
        env={"GH_TOKEN": "{{ env('GH_TOKEN') }}", "LOG_DIR": "/tmp/logs"},
        args=None,
        command="echo",
    )
    toolbox = SimpleNamespace(
        server_params=server_params,
        confirm=[],
        server_prompt=None,
        client_session_timeout=None,
    )
    available_tools = MagicMock()
    available_tools.get_toolbox.return_value = toolbox

    with caplog.at_level(logging.DEBUG):
        params = mcp_client_params(available_tools, ["pkg.tb"])

    # The resolved secret value never reaches the logs ...
    assert sentinel not in caplog.text
    # ... but the variable names are still logged for debuggability.
    assert "GH_TOKEN" in caplog.text
    assert "LOG_DIR" in caplog.text
    # Redaction is log-only: the real env (with the resolved secret) is still
    # passed through to the MCP server.
    assert params["pkg.tb"][0]["env"]["GH_TOKEN"] == sentinel


def test_env_names_returns_sorted_names_or_none():
    # Non-empty env -> sorted names; empty dict -> empty list; None -> None
    # (honouring the list[str] | None contract, never leaking values).
    assert _env_names({"B_VAR": "x", "A_VAR": "{{ env('A') }}"}) == ["A_VAR", "B_VAR"]
    assert _env_names({}) == []
    assert _env_names(None) is None


@pytest.mark.parametrize("kind", ["streamable", "sse"])
def test_mcp_client_params_resolves_remote_url_env(monkeypatch, kind):
    # A remote toolbox sources its endpoint from the environment. The url must
    # be env-templated the same way stdio args/env and headers already are, so
    # `url: "{{ env('CONTAINER_SHELL_URL') }}"` reaches the client resolved.
    monkeypatch.setenv("CONTAINER_SHELL_URL", "http://host.docker.internal:8765/mcp/")
    server_params = SimpleNamespace(
        kind=kind,
        reconnecting=False,
        url="{{ env('CONTAINER_SHELL_URL') }}",
        headers=None,
        optional_headers=None,
        timeout=None,
        command=None,
        env=None,
        args=None,
    )
    toolbox = SimpleNamespace(
        server_params=server_params,
        confirm=["container_shell_exec"],
        server_prompt=None,
        client_session_timeout=None,
    )
    available_tools = MagicMock()
    available_tools.get_toolbox.return_value = toolbox

    params = mcp_client_params(available_tools, ["pkg.remote"])

    assert params["pkg.remote"][0]["url"] == "http://host.docker.internal:8765/mcp/"
