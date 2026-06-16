# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for MCPNamespaceWrap."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from seclab_taskflow_agent.mcp_utils import MCPNamespaceWrap, compress_name


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
