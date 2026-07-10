# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""MCP client utilities.

Provides tool-name compression, namespace-aware MCP wrappers with
confirmation support, and toolbox parameter resolution.
"""

from __future__ import annotations

__all__ = [
    "COMPRESSED_NAME_LENGTH",
    "DEFAULT_MCP_CLIENT_SESSION_TIMEOUT",
    "MCPNamespaceWrap",
    "compress_name",
    "mcp_client_params",
]

import hashlib
import json
import logging
import os
import shutil
from typing import Any

from mcp.types import CallToolResult, TextContent

from .available_tools import AvailableTools
from .env_utils import swap_env

# Re-export transport classes and prompt builder so that existing
# ``from .mcp_utils import …`` statements continue to work.
from .mcp_prompt import mcp_system_prompt as mcp_system_prompt  # noqa: F401
from .mcp_transport import (  # noqa: F401
    AsyncDebugMCPServerStdio as AsyncDebugMCPServerStdio,
    ReconnectingMCPServerStdio as ReconnectingMCPServerStdio,
    StreamableMCPThread as StreamableMCPThread,
)

DEFAULT_MCP_CLIENT_SESSION_TIMEOUT: int = 120

# The OpenAI API rejects tool names longer than 64 characters.
# We hash long names down to this many hex characters.
COMPRESSED_NAME_LENGTH: int = 12


def compress_name(name: str) -> str:
    """Return a short hash of *name* to fit the OpenAI 64-char tool-name limit.

    Args:
        name: The original tool / toolbox name.

    Returns:
        A 12-character lowercase hex digest.
    """
    m = hashlib.sha256()
    m.update(name.encode("utf-8"))
    return m.hexdigest()[:COMPRESSED_NAME_LENGTH]


class MCPNamespaceWrap:
    """MCP client wrapper that prefixes tool names with a namespace hash.

    Also provides optional interactive confirmation before calling
    specific tools.

    Args:
        confirms: Tool names that require user confirmation.
        obj: The underlying MCP server/client object to wrap.
    """

    def __init__(self, confirms: list[str], obj: Any) -> None:
        self.confirms: list[str] = confirms
        self._obj: Any = obj
        self.namespace: str = compress_name(obj.name)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._obj, name)
        if callable(attr):
            match name:
                case "call_tool":
                    return self.call_tool
                case "list_tools":
                    return self.list_tools
                case _:
                    return attr
        return attr

    async def list_tools(self, *args: Any, **kwargs: Any) -> list[Any]:
        """List tools with namespace-prefixed names."""
        result = await self._obj.list_tools(*args, **kwargs)
        namespaced_tools: list[Any] = []
        for tool in result:
            tool_copy = tool.copy()
            tool_copy.name = f"{self.namespace}{tool.name}"
            namespaced_tools.append(tool_copy)
        return namespaced_tools

    async def list_tools_unfiltered(self) -> list[Any]:
        """List tools directly from the MCP session, namespace-prefixed.

        Bypasses any tool_filter configured on the wrapped openai-agents
        server (which would require ``run_context`` and ``agent`` arguments
        that aren't available when listing tools outside the openai-agents
        run loop -- e.g. when handing tools to a different SDK at build
        time).

        Prefixing is idempotent: if a tool's name already starts with this
        wrapper's namespace (e.g. because the underlying session returned a
        previously-namespaced object), the existing prefix is stripped
        before re-applying so calling this method multiple times never
        yields ``<ns><ns>name``.

        Raises ``RuntimeError`` if the underlying server has no active
        MCP session yet (caller should ensure the server is connected
        before calling this).
        """
        session = getattr(self._obj, "session", None)
        if session is None:
            raise RuntimeError(
                f"MCPNamespaceWrap({self._obj!r}): underlying server has no "
                "active MCP session; cannot list tools unfiltered"
            )
        result = await session.list_tools()
        namespaced_tools: list[Any] = []
        for tool in result.tools:
            tool_copy = tool.copy() if hasattr(tool, "copy") else tool
            # Idempotent: strip existing prefix before re-applying
            base_name = tool_copy.name.removeprefix(self.namespace)
            tool_copy.name = f"{self.namespace}{base_name}"
            namespaced_tools.append(tool_copy)
        return namespaced_tools

    def confirm_tool(self, tool_name: str, args: list[Any]) -> bool:
        """Interactively prompt the user for tool-call confirmation.

        Args:
            tool_name: The tool being invoked.
            args: Positional arguments to display.

        Returns:
            ``True`` if the user approved the call.
        """
        while True:
            yn = input(
                f"** 🤖❗ Allow tool call?: {tool_name}({','.join([json.dumps(arg) for arg in args])}) (yes/no): "
            )
            if yn in ["yes", "y"]:
                return True
            if yn in ["no", "n"]:
                return False

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        """Call a tool, stripping the namespace prefix and optionally confirming."""
        _args = list(args)
        tool_name: str = _args[0]
        tool_name = tool_name.removeprefix(self.namespace)
        # to run headless, just make confirms an empty list
        if self.confirms and tool_name in self.confirms:
            if not self.confirm_tool(tool_name, _args[1:]):
                result = CallToolResult(
                    content=[TextContent(type="text", text="Tool call not allowed.", annotations=None, meta=None)]
                )
                return result
        _args[0] = tool_name
        args = tuple(_args)
        result = await self._obj.call_tool(*args, **kwargs)
        return result


ClientParamsMap = dict[str, tuple[dict[str, Any], list[str], str | None, int | None]]


def _env_names(env: dict[str, Any] | None) -> list[str] | None:
    """Return only the environment variable names, for safe debug logging.

    Tool-call environments routinely carry credentials (e.g. ``GH_TOKEN``), so
    their values must never be written to logs. Callers log the names to keep
    debug output useful without leaking secrets.
    """
    return sorted(env) if env else env


def mcp_client_params(
    available_tools: AvailableTools,
    requested_toolboxes: list[str],
) -> ClientParamsMap:
    """Resolve toolbox configs into MCP server connection parameters.

    Args:
        available_tools: The tool registry that can look up toolbox configs.
        requested_toolboxes: Module paths of the toolboxes to resolve.

    Returns:
        A mapping from toolbox name to a tuple of
        ``(server_params, confirms, server_prompt, client_session_timeout)``.

    Raises:
        ValueError: If the transport kind is not supported.
        FileNotFoundError: If a streamable command cannot be found on ``$PATH``.
    """
    client_params: ClientParamsMap = {}
    for tb in requested_toolboxes:
        toolbox = available_tools.get_toolbox(tb)
        sp = toolbox.server_params
        kind: str = sp.kind
        reconnecting: bool = sp.reconnecting
        server_params: dict[str, Any] = {"kind": kind, "reconnecting": reconnecting}

        match kind:
            case "stdio":
                env = dict(sp.env) if sp.env else None
                args = list(sp.args) if sp.args else None
                logging.debug("Initializing toolbox: %s\nargs:\n%s\nenv names:\n%s\n", tb, args, _env_names(env))
                if env:
                    for k, v in list(env.items()):
                        try:
                            env[k] = swap_env(v)
                        except LookupError as e:
                            logging.critical(e)
                            logging.info("Assuming toolbox has default configuration available")
                            del env[k]
                    # Inherit proxy env vars from parent so HTTP clients
                    # work correctly inside network-isolated environments (AWF)
                    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
                                      "http_proxy", "https_proxy", "no_proxy"):
                        if proxy_var not in env and proxy_var in os.environ:
                            env[proxy_var] = os.environ[proxy_var]
                logging.debug("Tool call environment names: %s", _env_names(env))
                if args:
                    for i, v in enumerate(args):
                        args[i] = swap_env(v)
                logging.debug("Tool call args: %s", args)
                server_params["command"] = sp.command
                server_params["args"] = args
                server_params["env"] = env

            case "sse":
                headers = _resolve_headers(sp.headers, sp.optional_headers)
                server_params["url"] = sp.url
                server_params["headers"] = headers
                server_params["timeout"] = sp.timeout

            case "streamable":
                headers = _resolve_headers(sp.headers, sp.optional_headers)
                server_params["url"] = sp.url
                server_params["headers"] = headers
                server_params["timeout"] = sp.timeout

                if sp.command is not None:
                    env = dict(sp.env) if sp.env else None
                    args = list(sp.args) if sp.args else None
                    logging.debug("Initializing streamable toolbox: %s\nargs:\n%s\nenv names:\n%s\n", tb, args, _env_names(env))
                    exe = shutil.which(sp.command)
                    if exe is None:
                        raise FileNotFoundError(f"Could not resolve path to {sp.command}")
                    start_cmd = [exe]
                    if args:
                        for i, v in enumerate(args):
                            args[i] = swap_env(v)
                        start_cmd += args
                    server_params["command"] = start_cmd
                    if env:
                        for k, v in list(env.items()):
                            try:
                                env[k] = swap_env(v)
                            except LookupError as e:
                                logging.critical(e)
                                logging.info("Assuming toolbox has default configuration available")
                                del env[k]
                    server_params["env"] = env

            case _:
                raise ValueError(f"Unsupported MCP transport {kind}")

        client_params[tb] = (
            server_params,
            list(toolbox.confirm),
            toolbox.server_prompt,
            toolbox.client_session_timeout,
        )
    return client_params


def _resolve_headers(
    headers: dict[str, str] | None,
    optional_headers: dict[str, str] | None,
) -> dict[str, str] | None:
    """Expand env references in headers and merge required + optional.

    Required headers raise on missing env vars; optional headers are
    silently dropped when a referenced variable is absent.

    Args:
        headers: Header dict whose values may contain ``{{ env('…') }}``.
        optional_headers: Like *headers*, but missing env vars are tolerated.

    Returns:
        Merged header dict, or ``None`` if both inputs are ``None``.
    """
    resolved: dict[str, str] | None = None
    if headers:
        resolved = dict(headers)
        for k, v in resolved.items():
            resolved[k] = swap_env(v)
    resolved_optional: dict[str, str] | None = None
    if optional_headers:
        resolved_optional = dict(optional_headers)
        for k, v in list(resolved_optional.items()):
            try:
                resolved_optional[k] = swap_env(v)
            except LookupError:
                del resolved_optional[k]
    return _merge_headers(resolved, resolved_optional)


def _merge_headers(
    headers: dict[str, str] | None,
    optional_headers: dict[str, str] | None,
) -> dict[str, str] | None:
    """Merge required and optional header dicts.

    Args:
        headers: Required headers (may be ``None``).
        optional_headers: Optional headers (may be ``None``).

    Returns:
        Combined header dict, or ``None`` if both are ``None``.
    """
    if headers and optional_headers:
        headers.update(optional_headers)
        return headers
    return headers or optional_headers
