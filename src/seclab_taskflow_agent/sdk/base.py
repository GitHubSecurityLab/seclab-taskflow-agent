# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Backend-neutral types shared by every adapter."""

from __future__ import annotations

__all__ = [
    "AgentBackend",
    "AgentSpec",
    "MCPServerSpec",
    "StreamEvent",
    "TextDelta",
    "ToolEnd",
]

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, Union


@dataclass(frozen=True)
class TextDelta:
    """Incremental text chunk emitted while the model is generating."""

    text: str


@dataclass(frozen=True)
class ToolEnd:
    """A tool call has completed; ``text`` is the raw tool result."""

    tool_name: str
    text: str


StreamEvent = Union[TextDelta, ToolEnd]


@dataclass(frozen=True)
class MCPServerSpec:
    """Backend-neutral MCP server descriptor.

    The openai-agents adapter only needs the built native server
    handle that the runner stores under ``params["_native"]``. The
    Copilot SDK adapter consumes ``kind`` plus the transport keys
    (``command``/``args``/``env``/``url``/``headers``).
    """

    name: str
    kind: str  # "stdio" | "sse" | "streamable"
    params: dict[str, Any]


@dataclass
class AgentSpec:
    """Backend-neutral agent configuration."""

    name: str
    instructions: str
    model: str
    model_settings: dict[str, Any] = field(default_factory=dict)
    mcp_servers: list[MCPServerSpec] = field(default_factory=list)
    handoffs: list[AgentSpec] = field(default_factory=list)
    exclude_from_context: bool = False
    api_type: str | None = None
    endpoint: str | None = None
    token_env: str | None = None
    in_handoff_graph: bool = False
    blocked_tools: list[str] = field(default_factory=list)
    headless: bool = False


class AgentBackend(Protocol):
    """The contract every backend adapter implements."""

    name: str

    def validate(self, spec: AgentSpec) -> None:
        """Reject any spec field the backend cannot honour."""

    async def build(
        self,
        spec: AgentSpec,
        *,
        run_hooks: Any = None,
        agent_hooks: Any = None,
    ) -> Any:
        """Construct a backend-native agent from a neutral spec."""

    def run_streamed(
        self,
        agent: Any,
        prompt: str,
        *,
        max_turns: int,
    ) -> AsyncIterator[StreamEvent]:
        """Run the agent against *prompt*, yielding neutral stream events."""

    async def aclose(self, agent: Any) -> None:
        """Release any resources held by *agent*."""
