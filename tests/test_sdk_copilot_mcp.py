# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Offline tests for policy-enforcing Copilot MCP tools."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("copilot")

from copilot import ToolInvocation
from mcp.types import CallToolResult, ImageContent, ListToolsResult, TextContent, Tool

from seclab_taskflow_agent.mcp_transport import ReconnectingMCPServerStdio
from seclab_taskflow_agent.mcp_utils import MCPNamespaceWrap
from seclab_taskflow_agent.sdk.base import MCPServerSpec
from seclab_taskflow_agent.sdk.copilot_sdk.mcp import build_mcp_tools
from seclab_taskflow_agent.sdk.errors import BackendCapabilityError


def _server(*, confirms=()):
    transport = SimpleNamespace(
        name="toolbox",
        session=SimpleNamespace(
            list_tools=AsyncMock(
                return_value=ListToolsResult(
                    tools=[Tool(name="echo", description="Echo input", inputSchema={"type": "object"})],
                )
            )
        ),
        call_tool=AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="tool-output")],
            )
        ),
    )
    return MCPNamespaceWrap(list(confirms), transport), transport


async def _tools(server, *, blocked=(), exclude=False, kind="stdio"):
    return await build_mcp_tools(
        [MCPServerSpec(name="toolbox", kind=kind, params={"_native": server})],
        blocked_tools=list(blocked),
        exclude_from_context=exclude,
    )


@pytest.mark.parametrize("kind", ["stdio", "sse", "streamable"])
def test_registers_and_invokes_existing_mcp_connection(kind):
    server, transport = _server()

    async def _run():
        (tool,) = await _tools(server, kind=kind)
        assert tool.name == server.namespace + "echo"
        assert tool.description == "Echo input"
        assert tool.parameters == {"type": "object"}
        assert not tool.skip_permission
        assert not tool.is_terminal
        assert tool.defer == "never"
        return await tool.handler(ToolInvocation(tool_name=tool.name, arguments={"value": 42}))

    result = asyncio.run(_run())
    assert result.result_type == "success"
    assert result.text_result_for_llm == "tool-output"
    transport.call_tool.assert_awaited_once_with("echo", {"value": 42})


@pytest.mark.parametrize("namespaced", [False, True])
def test_blocked_tools_are_not_registered(namespaced):
    server, transport = _server()
    blocked = server.namespace + "echo" if namespaced else "echo"
    assert asyncio.run(_tools(server, blocked=[blocked])) == []
    transport.call_tool.assert_not_awaited()


@pytest.mark.parametrize("namespaced", [False, True])
def test_paginated_tools_are_filtered_and_invoked_through_the_wrapper(namespaced):
    server, transport = _server(confirms=["later"])
    server.confirm_tool = Mock(return_value=True)
    transport.session.list_tools.side_effect = [
        ListToolsResult(tools=[Tool(name="echo", inputSchema={})], nextCursor="next"),
        ListToolsResult(
            tools=[Tool(name="blocked", inputSchema={}), Tool(name="later", inputSchema={})],
        ),
    ]
    blocked = server.namespace + "blocked" if namespaced else "blocked"

    async def _run():
        tools = {tool.name.removeprefix(server.namespace): tool for tool in await _tools(server, blocked=[blocked])}
        assert set(tools) == {"echo", "later"}
        tool = tools["later"]
        return await tool.handler(ToolInvocation(tool_name=tool.name, arguments={"value": 42}))

    result = asyncio.run(_run())
    assert result.result_type == "success"
    server.confirm_tool.assert_called_once_with("later", [{"value": 42}])
    transport.call_tool.assert_awaited_once_with("later", {"value": 42})


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("exclude", [False, True])
def test_toolbox_confirmation_cannot_be_bypassed(approved, exclude):
    server, transport = _server(confirms=["echo"])
    server.confirm_tool = Mock(return_value=approved)

    async def _run():
        (tool,) = await _tools(server, exclude=exclude)
        return await tool.handler(ToolInvocation(tool_name=tool.name, arguments={"value": 42}))

    result = asyncio.run(_run())
    assert result.result_type == ("success" if approved else "failure")
    server.confirm_tool.assert_called_once_with("echo", [{"value": 42}])
    if approved:
        transport.call_tool.assert_awaited_once()
        assert result.text_result_for_llm == "tool-output"
    else:
        transport.call_tool.assert_not_awaited()
        assert "not allowed" in result.text_result_for_llm


def test_headless_wrapper_retains_explicit_confirmation_opt_out():
    server, transport = _server(confirms=[])
    server.confirm_tool = Mock(side_effect=AssertionError("unexpected confirmation"))

    async def _run():
        (tool,) = await _tools(server)
        await tool.handler(ToolInvocation(tool_name=tool.name))

    asyncio.run(_run())
    server.confirm_tool.assert_not_called()
    transport.call_tool.assert_awaited_once()


def test_missing_confirmation_fails_closed():
    server, transport = _server(confirms=["echo"])
    server.confirm_tool = Mock(side_effect=EOFError)

    async def _run():
        (tool,) = await _tools(server)
        return await tool.handler(ToolInvocation(tool_name=tool.name))

    result = asyncio.run(_run())
    assert result.result_type == "denied"
    transport.call_tool.assert_not_awaited()


def test_invocation_cannot_select_another_tool():
    server, transport = _server()

    async def _run():
        (tool,) = await _tools(server)
        return await tool.handler(ToolInvocation(tool_name="unregistered"))

    assert asyncio.run(_run()).result_type == "denied"
    transport.call_tool.assert_not_awaited()


def test_exclude_from_context_registers_terminal_tools():
    server, _ = _server()
    (tool,) = asyncio.run(_tools(server, exclude=True))
    assert tool.is_terminal


def test_mcp_errors_are_reported_as_tool_failures():
    server, transport = _server()
    transport.call_tool.return_value = CallToolResult(
        isError=True,
        content=[TextContent(type="text", text="invalid key")],
    )

    async def _run():
        (tool,) = await _tools(server)
        return await tool.handler(ToolInvocation(tool_name=tool.name))

    result = asyncio.run(_run())
    assert result.result_type == "failure"
    assert result.text_result_for_llm == "invalid key"


def test_preserves_structured_results_and_images():
    server, transport = _server()
    transport.call_tool.return_value = CallToolResult(
        content=[ImageContent(type="image", data="aW1hZ2U=", mimeType="image/png")],
        structuredContent={"value": [1, 2, 3]},
    )

    async def _run():
        (tool,) = await _tools(server)
        return await tool.handler(ToolInvocation(tool_name=tool.name))

    result = asyncio.run(_run())
    assert json.loads(result.text_result_for_llm) == {"value": [1, 2, 3]}
    assert result.binary_results_for_llm[0].data == "aW1hZ2U="
    assert result.binary_results_for_llm[0].mime_type == "image/png"


@pytest.mark.parametrize("structured", [{"items": [1, 2, 3]}, {}, {"result": [1, 2, 3]}])
def test_structured_payload_takes_precedence_over_text(structured):
    server, transport = _server()
    transport.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="Found 3 items")],
        structuredContent=structured,
    )

    async def _run():
        (tool,) = await _tools(server)
        return await tool.handler(ToolInvocation(tool_name=tool.name))

    result = asyncio.run(_run())
    assert json.loads(result.text_result_for_llm) == structured


@pytest.mark.parametrize(
    ("structured", "expected"),
    [
        ({"result": "[1, 2, 3]"}, [1, 2, 3]),
        ({"result": [1, 2, 3]}, [1, 2, 3]),
        ({"result": [1, 2, 3], "extra": True}, {"result": [1, 2, 3], "extra": True}),
    ],
)
def test_structured_fastmcp_envelopes_preserve_original_payloads(structured, expected):
    server, transport = _server()
    transport.session.list_tools.return_value.tools[0].outputSchema = {
        "type": "object",
        "properties": {"result": {}},
        "required": ["result"],
        "x-fastmcp-wrap-result": True,
    }
    transport.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="Found 3 items")],
        structuredContent=structured,
    )

    async def _run():
        (tool,) = await _tools(server)
        return await tool.handler(ToolInvocation(tool_name=tool.name))

    result = asyncio.run(_run())
    assert json.loads(result.text_result_for_llm) == expected


def test_raw_mcp_specs_cannot_bypass_runner_policy():
    with pytest.raises(BackendCapabilityError, match="connected by the taskflow runner"):
        asyncio.run(
            build_mcp_tools(
                [MCPServerSpec(name="unmanaged", kind="stdio", params={"command": "unused"})],
                blocked_tools=[],
                exclude_from_context=False,
            )
        )


@pytest.mark.parametrize("reconnecting", [False, True])
def test_memcache_round_trip_without_copilot_credentials(tmp_path, reconnecting):
    from agents.mcp import MCPServerStdio

    server_class = ReconnectingMCPServerStdio if reconnecting else MCPServerStdio

    async def _run():
        async with server_class(
            params={
                "command": sys.executable,
                "args": ["-m", "seclab_taskflow_agent.mcp_servers.memcache.memcache"],
                "env": {
                    "MEMCACHE_STATE_DIR": str(tmp_path),
                    "MEMCACHE_BACKEND": "dictionary_file",
                    "LOG_DIR": str(tmp_path),
                },
            },
            name="memcache",
        ) as transport:
            server = MCPNamespaceWrap([], transport)
            tools = {t.name.removeprefix(server.namespace): t for t in await _tools(server)}
            setter = tools["memcache_set_state"]
            result = await setter.handler(
                ToolInvocation(
                    tool_name=setter.name,
                    arguments={"key": "acc", "value": [1, 2, 3]},
                )
            )
            assert result.result_type == "success"
            getter = tools["memcache_get_state"]
            return await getter.handler(ToolInvocation(tool_name=getter.name, arguments={"key": "acc"}))

    async def _bounded():
        return await asyncio.wait_for(_run(), timeout=60)

    result = asyncio.run(_bounded())
    assert result.result_type == "success"
    assert json.loads(result.text_result_for_llm) == [1, 2, 3]
