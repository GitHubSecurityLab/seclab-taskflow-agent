# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""openai-agents backend adapter."""

from __future__ import annotations

__all__ = ["OpenAIAgentsBackend"]

import os
from collections.abc import AsyncIterator
from typing import Any

from agents.exceptions import AgentsException, MaxTurnsExceeded
from openai import APIConnectionError, APITimeoutError, BadRequestError, RateLimitError
from openai.types.responses import ResponseTextDeltaEvent

from ..base import AgentSpec, StreamEvent, TextDelta
from ..errors import (
    BackendBadRequestError,
    BackendMaxTurnsError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnexpectedError,
)


def _model_settings(spec: AgentSpec) -> dict[str, Any]:
    """Resolve effective ModelSettings kwargs for *spec*.

    Defaults match the runner's pre-abstraction behaviour: ``tool_choice``
    follows MCP availability, ``parallel_tool_calls`` is opt-in via
    ``MODEL_PARALLEL_TOOL_CALLS``, and ``temperature`` defaults to 0 for
    chat-completions runs (the Responses API rejects it). User-provided
    ``model_settings`` always wins.
    """
    has_tools = bool(spec.mcp_servers)
    settings: dict[str, Any] = {"tool_choice": "auto" if has_tools else None}
    settings["parallel_tool_calls"] = (
        bool(os.getenv("MODEL_PARALLEL_TOOL_CALLS")) if has_tools else None
    )
    model_temp = os.getenv("MODEL_TEMP")
    if model_temp is not None:
        settings["temperature"] = model_temp
    elif spec.api_type != "responses":
        settings["temperature"] = 0.0
    settings.update(spec.model_settings)
    return settings


class OpenAIAgentsBackend:
    """Adapter that drives the openai-agents SDK."""

    name = "openai_agents"

    def validate(self, spec: AgentSpec) -> None:
        # openai-agents accepts every field the YAML grammar exposes,
        # so there's nothing to reject up front.
        del spec

    async def build(
        self,
        spec: AgentSpec,
        *,
        run_hooks: Any = None,
        agent_hooks: Any = None,
    ) -> Any:
        from agents.agent import ModelSettings
        from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

        from ...agent import TaskAgent

        mcp_servers = [s.params["_native"] for s in spec.mcp_servers if "_native" in s.params]
        handoffs = [
            (await self.build(h, run_hooks=run_hooks, agent_hooks=agent_hooks)).agent
            for h in spec.handoffs
        ]
        instructions = (
            prompt_with_handoff_instructions(spec.instructions)
            if spec.in_handoff_graph
            else spec.instructions
        )
        return TaskAgent(
            name=spec.name,
            instructions=instructions,
            handoffs=handoffs,
            exclude_from_context=spec.exclude_from_context,
            mcp_servers=mcp_servers,
            model=spec.model,
            model_settings=ModelSettings(**_model_settings(spec)),
            api_type=spec.api_type or "chat_completions",
            endpoint=spec.endpoint,
            token=spec.token_env,
            run_hooks=run_hooks,
            agent_hooks=agent_hooks,
        )

    async def run_streamed(
        self,
        agent: Any,
        prompt: str,
        *,
        max_turns: int,
    ) -> AsyncIterator[StreamEvent]:
        result = None
        try:
            result = agent.run_streamed(prompt, max_turns=max_turns)
            try:
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(
                        event.data, ResponseTextDeltaEvent
                    ):
                        yield TextDelta(text=event.data.delta)
            finally:
                # aclose() on stream_events() throws GeneratorExit which
                # skips RunResultStreaming._cleanup_tasks(), so cancel
                # explicitly to keep _run_impl_task from leaking when
                # the consumer aborts the iterator (e.g. on idle timeout).
                if result is not None:
                    result.cancel()
        except MaxTurnsExceeded as exc:
            raise BackendMaxTurnsError(str(exc)) from exc
        except RateLimitError as exc:
            raise BackendRateLimitError(str(exc)) from exc
        except (APITimeoutError, APIConnectionError) as exc:
            raise BackendTimeoutError(str(exc)) from exc
        except BadRequestError as exc:
            raise BackendBadRequestError(str(exc)) from exc
        except AgentsException as exc:
            raise BackendUnexpectedError(str(exc)) from exc

    async def aclose(self, agent: Any) -> None:
        # MCP cleanup is owned by the runner; we only need to release
        # the per-agent AsyncOpenAI client so its httpx pool returns
        # any half-open sockets to the OS.
        if agent is not None:
            close = getattr(agent, "close", None)
            if close is not None:
                await close()
