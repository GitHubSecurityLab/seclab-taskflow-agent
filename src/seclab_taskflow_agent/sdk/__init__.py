# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Backend factory for the agent runner.

Three backends are supported: ``openai_agents`` (default), ``copilot_sdk``,
and ``anthropic_sdk``.  All three are always available because per-task
backend selection means any SDK may be needed at runtime.
"""

from __future__ import annotations

__all__ = [
    "AgentBackend",
    "AgentSpec",
    "MCPServerSpec",
    "StreamEvent",
    "TextDelta",
    "TokenUsage",
    "ToolEnd",
    "get_backend",
    "resolve_backend_name",
]

import os

from .base import (
    AgentBackend,
    AgentSpec,
    MCPServerSpec,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolEnd,
)

_ENV_VAR = "SECLAB_TASKFLOW_BACKEND"
_KNOWN = ("openai_agents", "copilot_sdk", "anthropic_sdk")
_BACKENDS: dict[str, AgentBackend] = {}


def get_backend(name: str) -> AgentBackend:
    """Return the backend adapter instance for *name*, importing it lazily."""
    if name not in _KNOWN:
        raise ValueError(f"Unknown backend {name!r}. Known: {_KNOWN}")
    if name not in _BACKENDS:
        if name == "openai_agents":
            from .openai_agents.backend import OpenAIAgentsBackend

            _BACKENDS[name] = OpenAIAgentsBackend()
        elif name == "copilot_sdk":
            from .copilot_sdk.backend import CopilotSDKBackend

            _BACKENDS[name] = CopilotSDKBackend()
        elif name == "anthropic_sdk":
            from .anthropic_sdk.backend import AnthropicSDKBackend

            _BACKENDS[name] = AnthropicSDKBackend()
        else:
            raise ValueError(f"No backend implementation for {name!r}")
    return _BACKENDS[name]


def resolve_backend_name(
    *,
    explicit: str | None = None,
    endpoint: str | None = None,
) -> str:
    """Pick the backend to use for a run.

    Precedence: ``explicit`` (from model config ``backend:`` field) >
    ``SECLAB_TASKFLOW_BACKEND`` env var > ``openai_agents``.

    Backend selection is always deterministic — there is no auto-detection
    based on endpoint URL.  Use ``backend: copilot_sdk`` or ``backend:
    anthropic_sdk`` in model config (or set
    ``SECLAB_TASKFLOW_BACKEND=<name>``) to opt in.

    The *endpoint* parameter is accepted for forward compatibility but
    is not used for backend selection.
    """
    del endpoint  # reserved for forward compat; not used for selection
    name = explicit or os.getenv(_ENV_VAR) or "openai_agents"
    if name not in _KNOWN:
        raise ValueError(f"Unknown backend {name!r}. Known: {_KNOWN}")
    return name
