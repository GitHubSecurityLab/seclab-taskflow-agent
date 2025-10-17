# https://openai.github.io/openai-agents-python/agents/
import os
import logging
from dotenv import load_dotenv
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from openai import AsyncOpenAI
from agents.agent import ModelSettings, ToolsToFinalOutputResult, FunctionToolResult
from agents.run import DEFAULT_MAX_TURNS
from agents.run import RunHooks
from agents import Agent, Runner, AgentHooks, RunHooks, result, function_tool, Tool, RunContextWrapper, TContext, OpenAIChatCompletionsModel, set_default_openai_client, set_default_openai_api, set_tracing_disabled

from .capi import COPILOT_INTEGRATION_ID, COPILOT_API_ENDPOINT

# grab our secrets from .env, this must be in .gitignore
load_dotenv()

match urlparse(COPILOT_API_ENDPOINT).netloc:
    case 'api.githubcopilot.com':
        default_model = 'gpt-4o'
    case 'models.github.ai':
        default_model = 'openai/gpt-4o'
    case _:
        raise ValueError(f"Unsupported Model Endpoint: {COPILOT_API_ENDPOINT}")

DEFAULT_MODEL = os.getenv('COPILOT_DEFAULT_MODEL', default=default_model)

# Run hooks monitor the entire lifetime of a runner, including across any Agent handoffs
class TaskRunHooks(RunHooks):
    def __init__(self,
                 on_agent_start: Callable | None = None,
                 on_agent_end: Callable | None = None,
                 on_tool_start: Callable | None = None,
                 on_tool_end: Callable | None = None):
        self._on_agent_start = on_agent_start
        self._on_agent_end = on_agent_end
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end

    async def on_agent_start(
            self,
            context: RunContextWrapper[TContext],
            agent: Agent[TContext]) -> None:
        logging.debug(f"TaskRunHooks on_agent_start: {agent.name}")
        if self._on_agent_start:
            await self._on_agent_start(context, agent)

    async def on_agent_end(
            self,
            context: RunContextWrapper[TContext],
            agent: Agent[TContext],
            output: Any) -> None:
        logging.debug(f"TaskRunHooks on_agent_end: {agent.name}")
        if self._on_agent_end:
            await self._on_agent_end(context, agent, output)

    async def on_tool_start(
            self,
            context: RunContextWrapper[TContext],
            agent: Agent[TContext],
            tool: Tool) -> None:
        logging.debug(f"TaskRunHooks on_tool_start: {tool.name}")
        if self._on_tool_start:
            await self._on_tool_start(context, agent, tool)

    async def on_tool_end(
            self,
            context: RunContextWrapper[TContext],
            agent: Agent[TContext],
            tool: Tool,
            result: str) -> None:
        logging.debug(f"TaskRunHooks on_tool_end: {tool.name} ")
        if self._on_tool_end:
            await self._on_tool_end(context, agent, tool, result)

# Agent hooks monitor the lifetime of a single agent, not across any Agent handoffs
class TaskAgentHooks(AgentHooks):
    def __init__(self,
                 on_handoff: Callable | None = None,
                 on_start: Callable | None = None,
                 on_end: Callable | None = None,
                 on_tool_start: Callable | None = None,
                 on_tool_end: Callable | None = None):
        self._on_handoff = on_handoff
        self._on_start = on_start
        self._on_end = on_end
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end

    async def on_handoff(
            self,
            context: RunContextWrapper[TContext],
            agent: Agent[TContext],
            source: Agent[TContext]) -> None:
        logging.debug(f"TaskAgentHooks on_handoff: {source.name} -> {agent.name}")
        if self._on_handoff:
            await self._on_handoff(context, agent, source)

    async def on_start(
            self,
            context: RunContextWrapper[TContext],
            agent: Agent[TContext]) -> None:
        logging.debug(f"TaskAgentHooks on_start: {agent.name}")
        if self._on_start:
            await self._on_start(context, agent)

    async def on_end(
            self,
            context: RunContextWrapper[TContext],
            agent: Agent[TContext],
            output: Any) -> None:
        logging.debug(f"TaskAgentHooks on_end: {agent.name}")
        if self._on_end:
            await self._on_end(context, agent, output)

    async def on_tool_start(
            self,
            context: RunContextWrapper[TContext],
            agent: Agent[TContext],
            tool: Tool) -> None:
        logging.debug(f"TaskAgentHooks on_tool_start: {tool.name}")
        if self._on_tool_start:
            await self._on_tool_start(context, agent, tool)

    async def on_tool_end(
            self,
            context: RunContextWrapper[TContext],
            agent: Agent[TContext],
            tool: Tool,
            result: str) -> None:
        logging.debug(f"TaskAgentHooks on_tool_end: {tool.name}")
        if self._on_tool_end:
            await self._on_tool_end(context, agent, tool, result)

class TaskAgent:
    def __init__(self,
                 name: str = 'TaskAgent',
                 instructions: str = '',
                 handoffs: list = [],
                 exclude_from_context: bool = False,
                 mcp_servers: dict = [],
                 model: str = DEFAULT_MODEL,
                 model_settings: ModelSettings | None = None,
                 run_hooks: TaskRunHooks | None = None,
                 agent_hooks: TaskAgentHooks | None = None):
        client = AsyncOpenAI(base_url=COPILOT_API_ENDPOINT,
                             api_key=os.getenv('COPILOT_TOKEN'),
                             default_headers={'Copilot-Integration-Id': COPILOT_INTEGRATION_ID})
        set_default_openai_client(client)
        # CAPI does not yet support the Responses API: https://github.com/github/copilot-api/issues/11185
        # as such we are implementing on chat completions for now
        set_default_openai_api("chat_completions")
        set_tracing_disabled(True)
        self.run_hooks = run_hooks or TaskRunHooks()
        # useful agent patterns:
        # openai/openai-agents-python/blob/main/examples/agent_patterns

        # when we want to exclude tool results from context, we receive results here instead of sending to LLM
        def _ToolsToFinalOutputFunction(context: RunContextWrapper[TContext],
                                        results: list[FunctionToolResult]) -> ToolsToFinalOutputResult:
            return ToolsToFinalOutputResult(True, "Excluding tool results from LLM context")

        self.agent = Agent(name=name,
                           instructions=instructions,
                           tool_use_behavior=_ToolsToFinalOutputFunction if exclude_from_context else 'run_llm_again',
                           model=OpenAIChatCompletionsModel(model=model, openai_client=client),
                           handoffs=handoffs,
                           mcp_servers=mcp_servers,
                           model_settings=model_settings or ModelSettings(),
                           hooks=agent_hooks or TaskAgentHooks())

    async def run(self, prompt: str, max_turns: int = DEFAULT_MAX_TURNS) -> result.RunResult:
        return await Runner.run(starting_agent=self.agent,
                                input=prompt,
                                max_turns=max_turns,
                                hooks=self.run_hooks)

    def run_streamed(self, prompt: str, max_turns: int = DEFAULT_MAX_TURNS) -> result.RunResultStreaming:
        return Runner.run_streamed(starting_agent=self.agent,
                                input=prompt,
                                max_turns=max_turns,
                                hooks=self.run_hooks)
