# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Translate neutral :class:`MCPServerSpec` values into Copilot SDK MCP configs."""

from __future__ import annotations

__all__ = ["build_mcp_config"]

import os
import shutil
from typing import Any

from ..base import MCPServerSpec


def _resolve_command(command: str) -> str:
    """Return an absolute path for *command*.

    The Copilot SDK spawns MCP servers in a fresh subprocess that does not
    inherit our active virtualenv, so a bare ``python`` resolves to the
    system interpreter and the toolbox import fails. Resolving against the
    runner's ``PATH`` (which has the venv prepended) gives the SDK an
    absolute path that points at the correct interpreter.
    """
    if not command or os.path.isabs(command):
        return command
    return shutil.which(command) or command


def _timeout_ms(params: dict[str, Any]) -> int | None:
    timeout_s = params.get("timeout")
    return int(timeout_s * 1000) if timeout_s else None


def _stdio_config(spec: MCPServerSpec) -> dict[str, Any]:
    p = spec.params
    config: dict[str, Any] = {
        "type": "stdio",
        "command": _resolve_command(p.get("command", "")),
        "args": list(p.get("args") or []),
        "tools": ["*"],  # session-level excluded_tools handles filtering
    }
    if p.get("env"):
        config["env"] = dict(p["env"])
    # The Copilot SDK does not inherit the runner's cwd by default, which
    # breaks toolboxes that resolve state directories from relative paths
    # (e.g. ``MEMCACHE_STATE_DIR=example_taskflow/``). Pin to the current
    # process directory unless the toolbox specifies one explicitly.
    config["cwd"] = p.get("cwd") or os.getcwd()
    timeout = _timeout_ms(p)
    if timeout:
        config["timeout"] = timeout
    return config


def _remote_config(spec: MCPServerSpec) -> dict[str, Any]:
    p = spec.params
    config: dict[str, Any] = {
        "type": "sse" if spec.kind == "sse" else "http",
        "url": p.get("url", ""),
        "tools": ["*"],
    }
    if p.get("headers"):
        config["headers"] = dict(p["headers"])
    timeout = _timeout_ms(p)
    if timeout:
        config["timeout"] = timeout
    return config


def build_mcp_config(specs: list[MCPServerSpec]) -> dict[str, dict[str, Any]]:
    """Return a ``mcp_servers`` mapping suitable for ``create_session``.

    Remote specs without a ``url`` (the runner is hosting the transport
    in-process) are skipped — the SDK cannot consume an in-process
    transport.
    """
    out: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.kind == "stdio":
            out[spec.name] = _stdio_config(spec)
        elif spec.kind in ("sse", "streamable") and spec.params.get("url"):
            out[spec.name] = _remote_config(spec)
    return out
