# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Expose runner-owned MCP tools through the public Copilot SDK tool API."""

from __future__ import annotations

__all__ = ["build_mcp_tools"]

import json
from typing import TYPE_CHECKING, Any

from ..base import MCPServerSpec
from ..errors import BackendCapabilityError

if TYPE_CHECKING:
    from copilot import Tool, ToolInvocation, ToolResult


def _tool_handler(server: Any, name: str, *, unwrap_result: bool):
    async def _invoke(invocation: ToolInvocation) -> ToolResult:
        from copilot import ToolBinaryResult, ToolResult
        from mcp.types import BlobResourceContents, EmbeddedResource, ImageContent, TextContent

        if invocation.tool_name != name:
            return ToolResult(result_type="denied", text_result_for_llm="Unknown taskflow tool")
        try:
            result = await server.call_tool(name, invocation.arguments or {})
        except (EOFError, OSError):
            return ToolResult(
                result_type="denied",
                text_result_for_llm="Required toolbox confirmation is unavailable",
            )

        texts: list[str] = []
        binaries: list[ToolBinaryResult] = []
        for content in result.content:
            if isinstance(content, TextContent):
                texts.append(content.text)
            elif isinstance(content, ImageContent):
                binaries.append(ToolBinaryResult(data=content.data, mime_type=content.mimeType, type="image"))
            elif isinstance(content, EmbeddedResource):
                resource = content.resource
                if isinstance(resource, BlobResourceContents):
                    binaries.append(
                        ToolBinaryResult(
                            data=resource.blob,
                            mime_type=resource.mimeType or "application/octet-stream",
                            type="resource",
                            description=str(resource.uri),
                        )
                    )
                else:
                    texts.append(resource.text)
            else:
                texts.append(content.model_dump_json(exclude_none=True))
        if result.structuredContent is not None:
            structured: Any = result.structuredContent
            # FastMCP explicitly marks envelopes for non-object return values.
            if unwrap_result and list(structured) == ["result"]:
                structured = structured["result"]
            texts = [structured if isinstance(structured, str) else json.dumps(structured)]
        return ToolResult(
            result_type="failure" if result.isError else "success",
            text_result_for_llm="".join(texts),
            binary_results_for_llm=binaries or None,
        )

    return _invoke


async def build_mcp_tools(
    specs: list[MCPServerSpec],
    *,
    blocked_tools: list[str],
    exclude_from_context: bool,
) -> list[Tool]:
    """Keep MCP execution behind SecLab's confirmation and transport wrappers.

    Registering the raw servers with Copilot as well would bypass those
    wrappers and start duplicate MCP connections. Terminal tools enforce
    ``exclude_from_context`` before the runtime can make another model call.
    """
    from copilot import Tool

    tools: list[Tool] = []
    blocked = set(blocked_tools)
    for spec in specs:
        server = spec.params.get("_native")
        if server is None:
            raise BackendCapabilityError(
                f"copilot_sdk: MCP server {spec.name!r} must be connected by the taskflow runner"
            )
        for tool in await server.list_tools_unfiltered():
            if tool.name in blocked or tool.name.removeprefix(server.namespace) in blocked:
                continue
            output_schema = getattr(tool, "outputSchema", None) or {}
            tools.append(
                Tool(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.inputSchema,
                    handler=_tool_handler(
                        server, tool.name, unwrap_result=output_schema.get("x-fastmcp-wrap-result") is True
                    ),
                    defer="never",
                    is_terminal=exclude_from_context,
                )
            )
    return tools
