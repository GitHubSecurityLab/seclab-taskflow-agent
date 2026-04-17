# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for MCP toolbox parameter resolution (``mcp_utils.mcp_client_params``)."""

from __future__ import annotations

from seclab_taskflow_agent.mcp_utils import mcp_client_params
from seclab_taskflow_agent.models import ServerParams, TaskflowHeader, ToolboxDocument


class _StubAvailableTools:
    """Minimal AvailableTools stand-in for ``mcp_client_params`` tests."""

    def __init__(self, toolboxes: dict[str, ToolboxDocument]) -> None:
        self._toolboxes = toolboxes

    def get_toolbox(self, name: str) -> ToolboxDocument:
        return self._toolboxes[name]


def _make_toolbox(server_params: ServerParams) -> ToolboxDocument:
    return ToolboxDocument(
        **{
            "seclab-taskflow-agent": TaskflowHeader(version="1.0", filetype="toolbox"),
            "server_params": server_params,
        }
    )


class TestStdioArgsHandling:
    """Regression tests for empty-list ``args`` coercion in stdio transport."""

    def test_empty_args_list_is_preserved_not_coerced_to_none(self) -> None:
        """An empty ``args: []`` must reach the client as ``[]``, not ``None``.

        ``StdioServerParameters`` (from the MCP SDK) rejects ``None`` for
        ``args`` with a Pydantic validation error. A binary that takes no
        arguments must therefore be invokable with ``args: []`` in the
        toolbox YAML.
        """
        tb_name = "stub.empty_args"
        toolbox = _make_toolbox(
            ServerParams(kind="stdio", command="some-binary", args=[], env=None)
        )
        params = mcp_client_params(_StubAvailableTools({tb_name: toolbox}), [tb_name])

        resolved = params[tb_name][0]
        assert resolved["command"] == "some-binary"
        assert resolved["args"] == [], (
            "Empty args list was coerced to None; "
            "this breaks StdioServerParameters validation."
        )

    def test_none_args_stays_none(self) -> None:
        tb_name = "stub.none_args"
        toolbox = _make_toolbox(
            ServerParams(kind="stdio", command="some-binary", args=None, env=None)
        )
        params = mcp_client_params(_StubAvailableTools({tb_name: toolbox}), [tb_name])

        assert params[tb_name][0]["args"] is None

    def test_non_empty_args_passes_through(self) -> None:
        tb_name = "stub.nonempty_args"
        toolbox = _make_toolbox(
            ServerParams(
                kind="stdio", command="some-binary", args=["--flag", "value"], env=None
            )
        )
        params = mcp_client_params(_StubAvailableTools({tb_name: toolbox}), [tb_name])

        assert params[tb_name][0]["args"] == ["--flag", "value"]
