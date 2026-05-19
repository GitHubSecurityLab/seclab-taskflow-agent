# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

# https://openai.github.io/openai-agents-python/agents/
import logging
import os
from collections.abc import Callable
from typing import Any

from agents import (
    Agent,
    AgentHooks,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    RunContextWrapper,
    RunHooks,
    Runner,
    TContext,
    Tool,
    result,
    set_tracing_disabled,
)
from agents.agent import FunctionToolResult, ModelSettings, ToolsToFinalOutputResult
from agents.run import DEFAULT_MAX_TURNS
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
import httpx

from .capi import get_AI_endpoint, get_AI_token, get_provider

__all__ = [
    "DEFAULT_MODEL",
    "TaskAgent",
    "TaskAgentHooks",
    "TaskRunHooks",
]

# grab our secrets from .env, this must be in .gitignore
load_dotenv(find_dotenv(usecwd=True))

api_endpoint = get_AI_endpoint()
_default_provider = get_provider(api_endpoint)
DEFAULT_MODEL = os.getenv("COPILOT_DEFAULT_MODEL", default=_default_provider.default_model)


class TaskRunHooks(RunHooks):
    """RunHooks that monitor the entire lifetime of a runner, including across Agent handoffs."""

    def __init__(
        self,
        on_agent_start: Callable | None = None,
        on_agent_end: Callable | None = None,
        on_tool_start: Callable | None = None,
        on_tool_end: Callable | None = None,
    ) -> None:
        """Initialize with optional callback functions for each lifecycle event."""
        self._on_agent_start = on_agent_start
        self._on_agent_end = on_agent_end
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end

    async def on_agent_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext]) -> None:
        """Called when an agent begins execution."""
        logging.debug(f"TaskRunHooks on_agent_start: {agent.name}")
        if self._on_agent_start:
            await self._on_agent_start(context, agent)

    async def on_agent_end(self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any) -> None:
        """Called when an agent finishes execution."""
        logging.debug(f"TaskRunHooks on_agent_end: {agent.name}")
        if self._on_agent_end:
            await self._on_agent_end(context, agent, output)

    async def on_tool_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool) -> None:
        """Called before a tool invocation begins."""
        logging.debug(f"TaskRunHooks on_tool_start: {tool.name}")
        if self._on_tool_start:
            await self._on_tool_start(context, agent, tool)

    async def on_tool_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str
    ) -> None:
        """Called after a tool invocation completes."""
        logging.debug(f"TaskRunHooks on_tool_end: {tool.name} ")
        if self._on_tool_end:
            await self._on_tool_end(context, agent, tool, result)


class TaskAgentHooks(AgentHooks):
    """AgentHooks that monitor the lifetime of a single agent, not across Agent handoffs."""

    def __init__(
        self,
        on_handoff: Callable | None = None,
        on_start: Callable | None = None,
        on_end: Callable | None = None,
        on_tool_start: Callable | None = None,
        on_tool_end: Callable | None = None,
    ) -> None:
        """Initialize with optional callback functions for each lifecycle event."""
        self._on_handoff = on_handoff
        self._on_start = on_start
        self._on_end = on_end
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end

    async def on_handoff(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], source: Agent[TContext]
    ) -> None:
        """Called when control is handed off from one agent to another."""
        logging.debug(f"TaskAgentHooks on_handoff: {source.name} -> {agent.name}")
        if self._on_handoff:
            await self._on_handoff(context, agent, source)

    async def on_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext]) -> None:
        """Called when the agent starts processing."""
        logging.debug(f"TaskAgentHooks on_start: {agent.name}")
        if self._on_start:
            await self._on_start(context, agent)

    async def on_end(self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any) -> None:
        """Called when the agent finishes processing."""
        logging.debug(f"TaskAgentHooks on_end: {agent.name}")
        if self._on_end:
            await self._on_end(context, agent, output)

    async def on_tool_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool) -> None:
        """Called before a tool invocation begins."""
        logging.debug(f"TaskAgentHooks on_tool_start: {tool.name}")
        if self._on_tool_start:
            await self._on_tool_start(context, agent, tool)

    async def on_tool_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str
    ) -> None:
        """Called after a tool invocation completes."""
        logging.debug(f"TaskAgentHooks on_tool_end: {tool.name}")
        if self._on_tool_end:
            await self._on_tool_end(context, agent, tool, result)


class TaskAgent:
    """High-level wrapper around the OpenAI Agent SDK.

    Configures the OpenAI client, creates an Agent with the given tools and
    model, and exposes ``run`` / ``run_streamed`` entry points.
    """

    def __init__(
        self,
        name: str = "TaskAgent",
        instructions: str = "",
        handoffs: list[Any] | None = None,
        exclude_from_context: bool = False,
        mcp_servers: list[Any] | None = None,
        model: str = DEFAULT_MODEL,
        model_settings: ModelSettings | None = None,
        api_type: str = "chat_completions",
        endpoint: str | None = None,
        token: str | None = None,
        run_hooks: TaskRunHooks | None = None,
        agent_hooks: TaskAgentHooks | None = None,
    ) -> None:
        """Create a TaskAgent with the specified configuration.

        Args:
            api_type: ``"chat_completions"`` or ``"responses"``.
            endpoint: Optional API endpoint URL override for this model.
            token: Optional env var name whose value is used as the API key.
        """
        # Resolve per-model endpoint and token, falling back to defaults
        resolved_endpoint = endpoint or api_endpoint
        if token:
            resolved_token = os.getenv(token, "")
            if not resolved_token:
                raise RuntimeError(f"Token env var {token!r} is not set")
        else:
            resolved_token = get_AI_token()

        # Only send provider-specific headers to matching endpoints
        provider = get_provider(resolved_endpoint)
        # httpx defaults to no read timeout, which lets a streaming run
        # block forever on a half-open TCP connection (CLOSE_WAIT). Pin
        # explicit per-phase timeouts so dead sockets surface as
        # APITimeoutError and our retry loop can recover.
        client = AsyncOpenAI(
            base_url=resolved_endpoint,
            api_key=resolved_token,
            default_headers=provider.extra_headers or None,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=60.0),
        )
        self._openai_client = client
        set_tracing_disabled(True)
        self.run_hooks = run_hooks or TaskRunHooks()

        # when we want to exclude tool results from context, we receive results here instead of sending to LLM
        def _ToolsToFinalOutputFunction(
            context: RunContextWrapper[TContext], results: list[FunctionToolResult]
        ) -> ToolsToFinalOutputResult:
            return ToolsToFinalOutputResult(True, "Excluding tool results from LLM context")

        # Select model class based on api_type
        if api_type == "responses":
            model_impl = OpenAIResponsesModel(model=model, openai_client=client)
        else:
            model_impl = OpenAIChatCompletionsModel(model=model, openai_client=client)

        self.agent = Agent(
            name=name,
            instructions=instructions,
            tool_use_behavior=_ToolsToFinalOutputFunction if exclude_from_context else "run_llm_again",
            model=model_impl,
            handoffs=handoffs or [],
            mcp_servers=mcp_servers or [],
            model_settings=model_settings or ModelSettings(),
            hooks=agent_hooks or TaskAgentHooks(),
        )

    async def close(self) -> None:
        """Release the underlying httpx connection pool.

        Dead CLOSE_WAIT sockets left in the pool can keep kqueue/epoll
        spinning on the event loop after the agent is otherwise done,
        so the runner calls this in its ``finally`` to free them.
        """
        if self._openai_client is not None:
            await self._openai_client.close()

    async def run(self, prompt: str, max_turns: int = DEFAULT_MAX_TURNS) -> result.RunResult:
        """Run the agent to completion and return the result."""
        return await Runner.run(starting_agent=self.agent, input=prompt, max_turns=max_turns, hooks=self.run_hooks)

    def run_streamed(self, prompt: str, max_turns: int = DEFAULT_MAX_TURNS) -> result.RunResultStreaming:
        """Run the agent with streaming output."""
        return Runner.run_streamed(starting_agent=self.agent, input=prompt, max_turns=max_turns, hooks=self.run_hooks)
