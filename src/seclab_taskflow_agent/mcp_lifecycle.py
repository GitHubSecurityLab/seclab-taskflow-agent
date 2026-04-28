# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""MCP server lifecycle management.

Handles connecting, running, and cleaning up MCP server instances
used during taskflow execution.
"""

from __future__ import annotations

__all__ = ["MCP_CLEANUP_TIMEOUT", "build_mcp_servers", "mcp_session_task"]

import asyncio
import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable

from agents.mcp import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp, create_static_tool_filter

from .mcp_transport import ReconnectingMCPServerStdio, StreamableMCPThread
from .mcp_utils import (
    DEFAULT_MCP_CLIENT_SESSION_TIMEOUT,
    MCPNamespaceWrap,
    mcp_client_params,
)

if TYPE_CHECKING:
    from .available_tools import AvailableTools

MCP_CLEANUP_TIMEOUT = 5


class MCPServerEntry:
    """A paired MCP server wrapper and optional local process."""

    __slots__ = ("server", "process", "name")

    def __init__(self, server: MCPNamespaceWrap, process: StreamableMCPThread | None = None, name: str = ""):
        self.server = server
        self.process = process
        self.name = name


# Type alias for a builder function
MCPServerBuilder = Callable[[str, dict[str, Any], Any, int, list[str]], MCPServerEntry]

MCP_TRANSPORT_REGISTRY: dict[str, MCPServerBuilder] = {}


def register_transport(kind: str) -> Callable[[MCPServerBuilder], MCPServerBuilder]:
    """Decorator to register an MCP transport builder."""
    def decorator(builder: MCPServerBuilder) -> MCPServerBuilder:
        MCP_TRANSPORT_REGISTRY[kind] = builder
        return builder
    return decorator


@register_transport("stdio")
def _build_stdio(tb: str, params: dict[str, Any], tool_filter: Any, client_session_timeout: int, confirms: list[str]) -> MCPServerEntry:
    if params.get("reconnecting", False):
        mcp_server = ReconnectingMCPServerStdio(
            name=tb,
            params=params,
            tool_filter=tool_filter,
            client_session_timeout_seconds=client_session_timeout,
            cache_tools_list=True,
        )
    else:
        mcp_server = MCPServerStdio(
            name=tb,
            params=params,
            tool_filter=tool_filter,
            client_session_timeout_seconds=client_session_timeout,
            cache_tools_list=True,
        )
    return MCPServerEntry(MCPNamespaceWrap(confirms, mcp_server), None, name=tb)


@register_transport("sse")
def _build_sse(tb: str, params: dict[str, Any], tool_filter: Any, client_session_timeout: int, confirms: list[str]) -> MCPServerEntry:
    mcp_server = MCPServerSse(
        name=tb,
        params=params,
        tool_filter=tool_filter,
        client_session_timeout_seconds=client_session_timeout,
    )
    return MCPServerEntry(MCPNamespaceWrap(confirms, mcp_server), None, name=tb)


@register_transport("streamable")
def _build_streamable(tb: str, params: dict[str, Any], tool_filter: Any, client_session_timeout: int, confirms: list[str]) -> MCPServerEntry:
    server_proc = None
    if "command" in params:

        def _print_out(line: str) -> None:
            logging.info(f"Streamable MCP Server stdout: {line}")

        def _print_err(line: str) -> None:
            logging.info(f"Streamable MCP Server stderr: {line}")

        server_proc = StreamableMCPThread(
            params["command"],
            url=params["url"],
            env=params["env"],
            on_output=_print_out,
            on_error=_print_err,
        )
    mcp_server = MCPServerStreamableHttp(
        name=tb,
        params=params,
        tool_filter=tool_filter,
        client_session_timeout_seconds=client_session_timeout,
    )
    return MCPServerEntry(MCPNamespaceWrap(confirms, mcp_server), server_proc, name=tb)


# Freeze the registry after all transports are registered to prevent
# accidental mutation in tests or at runtime.
MCP_TRANSPORT_REGISTRY = MappingProxyType(MCP_TRANSPORT_REGISTRY)


def build_mcp_servers(
    available_tools: AvailableTools,
    toolboxes: list[str],
    blocked_tools: list[str] | None = None,
    headless: bool = False,
) -> list[MCPServerEntry]:
    """Build MCP server instances for the given toolboxes.

    Args:
        available_tools: Tool registry for loading toolbox configs.
        toolboxes: List of toolbox module paths.
        blocked_tools: Tool names to block.
        headless: If True, skip all confirmation prompts.

    Returns:
        List of MCPServerEntry instances ready for connection.
    """
    tool_filter = create_static_tool_filter(blocked_tool_names=blocked_tools) if blocked_tools else None
    mcp_params = mcp_client_params(available_tools, toolboxes)
    entries: list[MCPServerEntry] = []

    for tb, (params, confirms, server_prompt, client_session_timeout) in mcp_params.items():
        if headless:
            confirms = []
        client_session_timeout = client_session_timeout or DEFAULT_MCP_CLIENT_SESSION_TIMEOUT
        kind = params.get("kind")
        if kind is None:
            raise ValueError(f"Missing 'kind' key in MCP params for toolbox {tb!r}")
        builder = MCP_TRANSPORT_REGISTRY.get(kind)
        if builder is None:
            raise ValueError(f"Unsupported MCP transport: {kind!r}")

        entry = builder(tb, params, tool_filter, client_session_timeout, confirms)
        entries.append(entry)

    return entries


async def mcp_session_task(
    entries: list[MCPServerEntry],
    connected: asyncio.Event,
    cleanup: asyncio.Event,
) -> None:
    """Background task that manages MCP server connect/cleanup lifecycle.

    Args:
        entries: MCP server entries to manage.
        connected: Event to signal when all servers are connected.
        cleanup: Event to wait on before cleaning up.
    """
    try:
        for entry in entries:
            logging.debug(f"Connecting mcp server: {entry.name}")
            if entry.process is not None:
                entry.process.start()
                await entry.process.async_wait_for_connection(poll_interval=0.1)
            await entry.server.connect()

        connected.set()
        await cleanup.wait()

        for entry in list(reversed(entries)):
            try:
                logging.debug(f"Starting cleanup for mcp server: {entry.name}")
                await entry.server.cleanup()
                logging.debug(f"Cleaned up mcp server: {entry.name}")
                if entry.process is not None:
                    entry.process.stop()
                    try:
                        await asyncio.to_thread(entry.process.join_and_raise)
                    except Exception as e:
                        logging.warning(f"Streamable mcp server process exception: {e}")
            except asyncio.CancelledError:
                logging.exception(f"Timeout on cleanup for mcp server: {entry.name}")
    except RuntimeError:
        logging.exception("RuntimeError in mcp session task")
    except asyncio.CancelledError:
        logging.exception("Timeout on main session task")
    finally:
        entries.clear()
