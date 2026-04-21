# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Taskflow execution engine.

Contains the core logic for deploying task agents, executing taskflows,
and managing the agent lifecycle. Extracted from the original monolithic
``__main__.py``.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_MAX_TURNS",
    "MAX_API_RETRY",
    "MAX_RATE_LIMIT_BACKOFF",
    "RATE_LIMIT_BACKOFF",
    "deploy_task_agents",
    "run_main",
]

import asyncio
import json
import logging
import os
import uuid
from typing import Any

import jinja2
from agents import Agent, RunContextWrapper, TContext, Tool
from agents.agent import ModelSettings
from agents.exceptions import AgentsException, MaxTurnsExceeded
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from openai import APIConnectionError, APITimeoutError, BadRequestError, RateLimitError
from openai.types.responses import ResponseTextDeltaEvent

from .agent import DEFAULT_MODEL, TaskAgent, TaskAgentHooks, TaskRunHooks
from .available_tools import AvailableTools
from .env_utils import TmpEnv
from .mcp_lifecycle import MCP_CLEANUP_TIMEOUT, build_mcp_servers, mcp_session_task
from .models import ModelConfigDocument, PersonalityDocument, TaskDefinition
from .mcp_prompt import mcp_system_prompt
from .mcp_utils import compress_name, mcp_client_params
from .render_utils import flush_async_output, render_model_output
from .shell_utils import shell_tool_call
from .template_utils import render_template

DEFAULT_MAX_TURNS = 50  # Maximum agent turns before forced termination
RATE_LIMIT_BACKOFF = 5  # Initial backoff in seconds after a rate-limit response
MAX_RATE_LIMIT_BACKOFF = 120  # Maximum backoff cap in seconds for rate-limit retries
MAX_API_RETRY = 5  # Maximum number of consecutive API error retries
TASK_RETRY_LIMIT = 3  # Maximum retry attempts for a failed task
TASK_RETRY_BACKOFF = 10  # Initial backoff in seconds between task retries


def _resolve_model_config(
    available_tools: AvailableTools,
    model_config_ref: str,
) -> tuple[list[str], dict[str, str], dict[str, dict[str, Any]], str]:
    """Load and validate the model configuration file.

    Args:
        available_tools: Tool registry used to load the config file.
        model_config_ref: Reference name for the model config document.

    Returns:
        A tuple of (model_keys, model_dict, models_params, api_type) where
        model_keys is the list of logical model names, model_dict maps them
        to provider model IDs, models_params holds per-model settings, and
        api_type is ``"chat_completions"`` or ``"responses"``.

    Raises:
        ValueError: If the config file has structural problems.
    """
    m_config: ModelConfigDocument = available_tools.get_model_config(model_config_ref)
    model_dict: dict[str, str] = m_config.models or {}
    model_keys: list[str] = list(model_dict.keys())
    models_params: dict[str, dict[str, Any]] = m_config.model_settings or {}
    unknown = set(models_params) - set(model_keys)
    if unknown:
        raise ValueError(
            f"Settings section of model_config file {model_config_ref} contains models not in the model section: {unknown}"
        )
    return model_keys, model_dict, models_params, m_config.api_type


def _merge_reusable_task(
    available_tools: AvailableTools,
    task: TaskDefinition,
) -> TaskDefinition:
    """Merge a reusable taskflow into the current task definition.

    Args:
        available_tools: Tool registry used to load the reusable taskflow.
        task: Current task whose ``uses`` field references a reusable taskflow.

    Returns:
        A new TaskDefinition with parent defaults filled in where the current
        task uses its own defaults.

    Raises:
        ValueError: If the reusable taskflow is missing or has more than 1 task.
    """
    reusable_doc = available_tools.get_taskflow(task.uses)
    if reusable_doc is None:
        raise ValueError(f"No such reusable taskflow: {task.uses}")
    if len(reusable_doc.taskflow) > 1:
        raise ValueError("Reusable taskflows can only contain 1 task")
    parent_task = reusable_doc.taskflow[0].task
    merged: dict[str, Any] = parent_task.model_dump(by_alias=True, exclude_defaults=True)
    current: dict[str, Any] = task.model_dump(by_alias=True, exclude_defaults=True)
    merged.update(current)
    return TaskDefinition.model_validate(merged)


def _resolve_task_model(
    task: TaskDefinition,
    model_keys: list[str],
    model_dict: dict[str, str],
    models_params: dict[str, dict[str, Any]],
    default_api_type: str = "chat_completions",
) -> tuple[str, dict[str, Any], str, str | None, str | None]:
    """Resolve the final model name, settings, and per-model overrides.

    Returns:
        A tuple of ``(model_id, model_settings, api_type, endpoint, token)``
        where *endpoint* and *token* are ``None`` when not overridden.

    Raises:
        ValueError: If task-level model_settings is not a dictionary.
    """
    logical_name: str = task.model or DEFAULT_MODEL
    model_settings: dict[str, Any] = {}
    api_type: str = default_api_type
    endpoint: str | None = None
    token: str | None = None

    if logical_name in model_keys:
        if logical_name in models_params:
            model_settings = models_params[logical_name].copy()
        logical_name = model_dict[logical_name]

    # Extract engine-level keys before merging task settings
    api_type = model_settings.pop("api_type", api_type)
    endpoint = model_settings.pop("endpoint", None)
    token = model_settings.pop("token", None)

    task_model_settings: dict[str, Any] | Any = task.model_settings or {}
    if not isinstance(task_model_settings, dict):
        raise ValueError(f"model_settings in task {task.name or ''} needs to be a dictionary")

    # Task-level overrides can also set engine keys
    task_settings = dict(task_model_settings)
    api_type = task_settings.pop("api_type", api_type)
    endpoint = task_settings.pop("endpoint", endpoint)
    token = task_settings.pop("token", token)

    model_settings.update(task_settings)
    return logical_name, model_settings, api_type, endpoint, token


async def _build_prompts_to_run(
    task_prompt: str,
    repeat_prompt: bool,
    last_mcp_tool_results: list[str],
    available_tools: AvailableTools,
    global_variables: dict[str, Any],
    inputs: dict[str, Any],
) -> list[str]:
    """Build the list of prompts to execute for a task.

    For regular tasks the list contains a single rendered prompt.  When
    ``repeat_prompt`` is enabled, the last MCP tool result is parsed as an
    iterable and a prompt is rendered for each element.

    Args:
        task_prompt: The raw or pre-rendered prompt template string.
        repeat_prompt: Whether to expand prompts over MCP tool results.
        last_mcp_tool_results: Mutable list of prior MCP tool result strings.
        available_tools: Tool registry (passed through to template rendering).
        global_variables: Global template variables.
        inputs: Task-level input variables.

    Returns:
        List of rendered prompt strings to execute.

    Raises:
        ValueError: If the last MCP result is missing or not valid JSON.
    """
    prompts_to_run: list[str] = []
    if repeat_prompt:
        if "result" not in task_prompt.lower():
            logging.warning("repeat_prompt enabled but no {{ result }} in prompt")
        try:
            last_result = json.loads(last_mcp_tool_results[-1])
        except IndexError:
            logging.critical("No last MCP tool result available")
            raise
        except json.JSONDecodeError as exc:
            logging.critical(f"Could not parse tool result as JSON: {last_mcp_tool_results[-1][:200]}")
            raise ValueError("Tool result is not valid JSON") from exc

        text = last_result.get("text", "")
        try:
            iterable_result = json.loads(text)
        except json.JSONDecodeError as exc:
            logging.critical(f"Could not parse result text: {text}")
            raise ValueError("Result text is not valid JSON") from exc
        try:
            iter(iterable_result)
        except TypeError:
            logging.critical("Last MCP tool result is not iterable")
            raise

        if not iterable_result:
            await render_model_output("** 🤖❗MCP tool result iterable is empty!\n")
        else:
            logging.debug(f"Rendering templated prompts for results: {iterable_result}")
            for value in iterable_result:
                try:
                    rendered_prompt = render_template(
                        template_str=task_prompt,
                        available_tools=available_tools,
                        globals_dict=global_variables,
                        inputs_dict=inputs,
                        result_value=value,
                    )
                    prompts_to_run.append(rendered_prompt)
                except jinja2.TemplateError as e:
                    logging.error(f"Error rendering template for result {value}: {e}")
                    raise ValueError(f"Template rendering failed: {e}")

        # Consume only after all prompts rendered successfully so that
        # the result remains available for retry/resume on failure.
        last_mcp_tool_results.pop()
    else:
        prompts_to_run.append(task_prompt)
    return prompts_to_run


async def deploy_task_agents(
    available_tools: AvailableTools,
    agents: dict[str, PersonalityDocument],
    prompt: str,
    *,
    async_task: bool = False,
    toolboxes_override: list[str] | None = None,
    blocked_tools: list[str] | None = None,
    headless: bool = False,
    exclude_from_context: bool = False,
    max_turns: int = DEFAULT_MAX_TURNS,
    model: str = DEFAULT_MODEL,
    model_par: dict[str, Any] | None = None,
    api_type: str = "chat_completions",
    endpoint: str | None = None,
    token: str | None = None,
    run_hooks: TaskRunHooks | None = None,
    agent_hooks: TaskAgentHooks | None = None,
) -> bool:
    """Deploy and run task agents with MCP servers.

    Args:
        available_tools: Tool registry.
        agents: Mapping of agent name -> PersonalityDocument.
        prompt: User prompt to execute.
        api_type: OpenAI API type -- ``"chat_completions"`` or ``"responses"``.
        endpoint: Optional per-model API endpoint URL override.
        token: Optional env var name to resolve as the API token.

    Returns:
        True if the task completed successfully.
    """
    model_par = model_par or {}
    toolboxes_override = toolboxes_override or []
    blocked_tools = blocked_tools or []

    task_id = str(uuid.uuid4())
    await render_model_output(f"** 🤖💪 Deploying Task Flow Agent(s): {list(agents.keys())}\n")
    await render_model_output(f"** 🤖💪 Task ID : {task_id}\n")
    await render_model_output(f"** 🤖💪 Model   : {model}{', params: ' + str(model_par) if model_par else ''}\n")
    if endpoint:
        await render_model_output(f"** 🤖💪 Endpoint: {endpoint}\n")

    # Resolve toolboxes from personality definitions or override
    toolboxes: list[str] = []
    if toolboxes_override:
        toolboxes = toolboxes_override
    else:
        for personality in agents.values():
            for tb in personality.toolboxes:
                if tb not in toolboxes:
                    toolboxes.append(tb)

    # Model settings
    parallel_tool_calls = bool(os.getenv("MODEL_PARALLEL_TOOL_CALLS"))
    model_params: dict[str, Any] = {
        "tool_choice": "auto" if toolboxes else None,
        "parallel_tool_calls": parallel_tool_calls if toolboxes else None,
    }
    # Only inject a default temperature for chat_completions; the responses
    # API rejects unsupported parameters.  MODEL_TEMP env override applies
    # to both API types.
    model_temp = os.getenv("MODEL_TEMP")
    if model_temp is not None:
        model_params["temperature"] = model_temp
    elif api_type != "responses":
        model_params["temperature"] = 0.0
    model_params.update(model_par)
    model_settings = ModelSettings(**model_params)

    # Build MCP servers and collect server prompts
    entries = build_mcp_servers(available_tools, toolboxes, blocked_tools, headless)
    mcp_params = mcp_client_params(available_tools, toolboxes)
    server_prompts = [sp for _, (_, _, sp, _) in mcp_params.items()]

    # Connect MCP servers
    servers_connected = asyncio.Event()
    start_cleanup = asyncio.Event()
    mcp_sessions = asyncio.create_task(mcp_session_task(entries, servers_connected, start_cleanup))

    await servers_connected.wait()
    logging.debug("All mcp servers are connected!")

    try:
        important_guidelines = [
            "Do not prompt the user with questions.",
            "Run tasks until a final result is available.",
            "Ensure responses are based on the latest information from available tools.",
            "Run tools sequentially, wait until one tool has completed before calling the next.",
        ]

        # Create handoff agents from additional personalities
        handoffs = []
        agent_names = list(agents.keys())
        for handoff_name in agent_names[1:]:
            personality = agents[handoff_name]
            handoffs.append(
                TaskAgent(
                    name=compress_name(handoff_name),
                    instructions=prompt_with_handoff_instructions(
                        mcp_system_prompt(
                            personality.personality,
                            personality.task,
                            server_prompts=server_prompts,
                            important_guidelines=important_guidelines,
                        )
                    ),
                    handoffs=[],
                    exclude_from_context=exclude_from_context,
                    mcp_servers=[e.server for e in entries],
                    model=model,
                    model_settings=model_settings,
                    api_type=api_type,
                    endpoint=endpoint,
                    token=token,
                    run_hooks=run_hooks,
                    agent_hooks=agent_hooks,
                ).agent
            )

        # Create primary agent
        primary_name = agent_names[0]
        primary_personality = agents[primary_name]
        system_prompt = mcp_system_prompt(
            primary_personality.personality,
            primary_personality.task,
            server_prompts=server_prompts,
            important_guidelines=important_guidelines,
        )
        agent0 = TaskAgent(
            name=primary_name,
            instructions=prompt_with_handoff_instructions(system_prompt) if handoffs else system_prompt,
            handoffs=handoffs,
            exclude_from_context=exclude_from_context,
            mcp_servers=[e.server for e in entries],
            model=model,
            model_settings=model_settings,
            api_type=api_type,
            endpoint=endpoint,
            token=token,
            run_hooks=run_hooks,
            agent_hooks=agent_hooks,
        )

        try:
            complete = False

            async def _run_streamed() -> None:
                max_retry = MAX_API_RETRY
                rate_limit_backoff = RATE_LIMIT_BACKOFF
                while rate_limit_backoff:
                    try:
                        result = agent0.run_streamed(prompt, max_turns=max_turns)
                        async for event in result.stream_events():
                            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                                await render_model_output(event.data.delta, async_task=async_task, task_id=task_id)
                        await render_model_output("\n\n", async_task=async_task, task_id=task_id)
                        return
                    except APITimeoutError:
                        if not max_retry:
                            logging.exception("Max retries for APITimeoutError reached")
                            raise
                        max_retry -= 1
                    except RateLimitError:
                        if rate_limit_backoff == MAX_RATE_LIMIT_BACKOFF:
                            raise APITimeoutError("Max rate limit backoff reached")
                        if rate_limit_backoff > MAX_RATE_LIMIT_BACKOFF:
                            rate_limit_backoff = MAX_RATE_LIMIT_BACKOFF
                        else:
                            rate_limit_backoff += rate_limit_backoff
                        logging.exception(f"Hit rate limit ... holding for {rate_limit_backoff}")
                        await asyncio.sleep(rate_limit_backoff)

            await _run_streamed()
            complete = True

        except MaxTurnsExceeded as e:
            await render_model_output(f"** 🤖❗ Max Turns Reached: {e}\n", async_task=async_task, task_id=task_id)
            logging.exception(f"Exceeded max_turns: {max_turns}")
        except AgentsException as e:
            await render_model_output(f"** 🤖❗ Agent Exception: {e}\n", async_task=async_task, task_id=task_id)
            logging.exception("Agent Exception")
        except BadRequestError as e:
            await render_model_output(f"** 🤖❗ Request Error: {e}\n", async_task=async_task, task_id=task_id)
            logging.exception("Bad Request")
        except APITimeoutError as e:
            await render_model_output(f"** 🤖❗ Timeout Error: {e}\n", async_task=async_task, task_id=task_id)
            logging.exception("API Timeout")

        if async_task:
            await flush_async_output(task_id)

        return complete

    finally:
        start_cleanup.set()
        cleanup_attempts_left = len(entries)
        while cleanup_attempts_left and entries:
            try:
                cleanup_attempts_left -= 1
                await asyncio.wait_for(mcp_sessions, timeout=MCP_CLEANUP_TIMEOUT)
            except asyncio.TimeoutError:
                continue
            except Exception:
                logging.exception("Exception in mcp server cleanup task")


async def run_main(
    available_tools: AvailableTools,
    personality_path: str | None,
    taskflow_path: str | None,
    cli_globals: dict[str, str],
    prompt: str | None,
    resume_session_id: str | None = None,
) -> None:
    """Main entry point for taskflow/personality execution.

    Args:
        available_tools: Tool registry.
        personality_path: Personality module path, or None.
        taskflow_path: Taskflow module path, or None.
        cli_globals: Global variables from CLI.
        prompt: User prompt text.
        resume_session_id: Session ID to resume from a checkpoint.
    """
    from .session import TaskflowSession

    last_mcp_tool_results: list[str] = []

    async def on_tool_end_hook(context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str) -> None:
        last_mcp_tool_results.append(result)

    async def on_tool_start_hook(context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool) -> None:
        await render_model_output(f"\n** 🤖🛠️ Tool Call: {tool.name}\n")

    async def on_handoff_hook(context: RunContextWrapper[TContext], agent: Agent[TContext], source: Agent[TContext]) -> None:
        await render_model_output(f"\n** 🤖🤝 Agent Handoff: {source.name} -> {agent.name}\n")

    if personality_path:
        personality = available_tools.get_personality(personality_path)
        await deploy_task_agents(
            available_tools,
            {personality_path: personality},
            prompt or "",
            run_hooks=TaskRunHooks(on_tool_end=on_tool_end_hook, on_tool_start=on_tool_start_hook),
        )

    if taskflow_path or resume_session_id:
        # Handle session resume
        session: TaskflowSession | None = None
        if resume_session_id:
            session = TaskflowSession.load(resume_session_id)
            if session.finished:
                await render_model_output(f"** 🤖✅ Session {resume_session_id} already completed\n")
                return
            taskflow_path = session.taskflow_path
            cli_globals = session.cli_globals
            prompt = session.prompt
            last_mcp_tool_results = list(session.last_tool_results)
            await render_model_output(
                f"** 🤖🔄 Resuming session {resume_session_id} from task {session.next_task_index}\n"
            )

        taskflow_doc = available_tools.get_taskflow(taskflow_path)
        await render_model_output(f"** 🤖💪 Running Task Flow: {taskflow_path}\n")

        # Resolve global variables (file defaults + CLI overrides)
        global_variables = dict(taskflow_doc.globals or {})
        if cli_globals:
            global_variables.update(cli_globals)

        # Resolve model config
        model_config_ref = taskflow_doc.model_config_ref
        model_keys: list[str] = []
        model_dict: dict[str, str] = {}
        models_params: dict[str, dict[str, Any]] = {}
        api_type: str = "chat_completions"
        if model_config_ref:
            model_keys, model_dict, models_params, api_type = _resolve_model_config(available_tools, model_config_ref)

        # Create session if this is a new run (not personality mode)
        if session is None:
            session = TaskflowSession(
                taskflow_path=taskflow_path,
                cli_globals=cli_globals,
                prompt=prompt or "",
                total_tasks=len(taskflow_doc.taskflow),
            )
            session.save()
            await render_model_output(f"** 🤖📋 Session: {session.session_id}\n")

        for task_index, task_wrapper in enumerate(taskflow_doc.taskflow):
            # Skip already-completed tasks on resume
            if task_index < session.next_task_index:
                await render_model_output(
                    f"** 🤖⏭️ Skipping completed task {task_index}\n"
                )
                continue

            task = task_wrapper.task

            # Reusable taskflow support: merge parent defaults into current task
            if task.uses:
                task = _merge_reusable_task(available_tools, task)

            # Resolve model (name, settings, api_type, optional endpoint/token)
            model, model_settings, task_api_type, task_endpoint, task_token = _resolve_task_model(
                task, model_keys, model_dict, models_params, default_api_type=api_type,
            )

            # Read task fields via typed attributes
            agents_list = task.agents or []
            headless = task.headless
            blocked_tools = task.blocked_tools or []
            run = task.run or ""
            inputs = task.inputs or {}
            task_prompt = task.user_prompt or ""
            if run and task_prompt:
                raise ValueError("shell task and prompt task are mutually exclusive!")
            must_complete = task.must_complete
            max_turns = task.max_steps or DEFAULT_MAX_TURNS
            toolboxes_override = task.toolboxes or []
            env = task.env or {}
            repeat_prompt = task.repeat_prompt
            exclude_from_context = task.exclude_from_context
            async_task = task.async_task
            max_concurrent_tasks = task.async_limit

            # Render prompt template (skip if repeat_prompt — result not yet available)
            if task_prompt and not repeat_prompt:
                try:
                    task_prompt = render_template(
                        template_str=task_prompt,
                        available_tools=available_tools,
                        globals_dict=global_variables,
                        inputs_dict=inputs,
                    )
                except jinja2.TemplateError as e:
                    logging.error(f"Template rendering error: {e}")
                    raise ValueError(f"Failed to render prompt template: {e}") from e

            with TmpEnv(env, context={"globals": global_variables}):
                prompts_to_run: list[str] = await _build_prompts_to_run(
                    task_prompt, repeat_prompt, last_mcp_tool_results,
                    available_tools, global_variables, inputs,
                )

                async def run_prompts(async_task: bool = False, max_concurrent_tasks: int = 5) -> bool:
                    if run:
                        await render_model_output("** 🤖🐚 Executing Shell Task\n")
                        try:
                            result = shell_tool_call(run).content[0].model_dump_json()
                            last_mcp_tool_results.append(result)
                            return True
                        except RuntimeError as e:
                            await render_model_output(f"** 🤖❗ Shell Task Exception: {e}\n")
                            logging.exception("Shell task error")
                            return False

                    tasks: list[Any] = []
                    task_results: list[Any] = []
                    semaphore = asyncio.Semaphore(max_concurrent_tasks)
                    for p_prompt in prompts_to_run:
                        resolved_agents: dict[str, Any] = {}
                        current_agents = list(agents_list)
                        if not current_agents:
                            from .prompt_parser import parse_prompt_args
                            p_val, _, _, _, p_prompt, _ = parse_prompt_args(available_tools, p_prompt)
                            if p_val:
                                current_agents.append(p_val)
                        for agent_name in current_agents:
                            personality = available_tools.get_personality(agent_name)
                            if personality is None:
                                raise ValueError(f"No such personality: {agent_name}")
                            resolved_agents[agent_name] = personality

                        if not resolved_agents:
                            raise ValueError(
                                "No agents resolved for this task. "
                                "Specify a personality with -p or provide an agents list."
                            )

                        async def _deploy(ra: dict, pp: str) -> bool:
                            async with semaphore:
                                return await deploy_task_agents(
                                    available_tools,
                                    ra,
                                    pp,
                                    async_task=async_task,
                                    toolboxes_override=toolboxes_override,
                                    blocked_tools=blocked_tools,
                                    headless=headless,
                                    exclude_from_context=exclude_from_context,
                                    max_turns=max_turns,
                                    run_hooks=TaskRunHooks(
                                        on_tool_end=on_tool_end_hook, on_tool_start=on_tool_start_hook
                                    ),
                                    model=model,
                                    model_par=model_settings,
                                    api_type=task_api_type,
                                    endpoint=task_endpoint,
                                    token=task_token,
                                    agent_hooks=TaskAgentHooks(on_handoff=on_handoff_hook),
                                )

                        task_coroutine = _deploy(resolved_agents, p_prompt)

                        if not async_task:
                            result = await task_coroutine
                            task_results.append(result)
                        else:
                            tasks.append(task_coroutine)

                    if async_task:
                        task_results = await asyncio.gather(*tasks, return_exceptions=True)

                    complete = True
                    for result in task_results:
                        if isinstance(result, Exception):
                            logging.error(f"Caught exception in Gather: {result}")
                            result = False
                        complete = result and complete
                    return complete

                # Execute the task with auto-retry on transient failures.
                # Only retry on network/API errors — deterministic failures
                # and errors after side-effectful work should not be retried
                # blindly (e.g. repeat_prompt tasks may have already written
                # data to external systems).
                task_name = task.name or f"task-{task_index}"
                task_complete = False
                last_task_error: BaseException | None = None

                for attempt in range(TASK_RETRY_LIMIT):
                    try:
                        task_complete = await run_prompts(
                            async_task=async_task,
                            max_concurrent_tasks=max_concurrent_tasks,
                        )
                        last_task_error = None
                        break
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError) as exc:
                        last_task_error = exc
                        remaining = TASK_RETRY_LIMIT - attempt - 1
                        if remaining > 0:
                            backoff = TASK_RETRY_BACKOFF * (attempt + 1)
                            await render_model_output(
                                f"** 🤖🔄 Task {task_name!r} failed: {exc}\n"
                                f"** 🤖🔄 Retrying in {backoff}s ({remaining} attempts left)\n"
                            )
                            logging.warning(f"Task {task_name!r} attempt {attempt + 1} failed: {exc}")
                            await asyncio.sleep(backoff)
                        else:
                            logging.error(f"Task {task_name!r} failed after {TASK_RETRY_LIMIT} attempts: {exc}")
                    except Exception as exc:
                        last_task_error = exc
                        logging.error(f"Task {task_name!r} failed (non-retriable): {exc}")
                        break

                # If all retries exhausted with an exception, save and re-raise
                if last_task_error is not None:
                    session.mark_failed(f"Task {task_name!r}: {last_task_error}")
                    await render_model_output(
                        f"** 🤖💾 Session saved: {session.session_id}\n"
                        f"** 🤖💡 Resume with: --resume {session.session_id}\n"
                    )
                    raise last_task_error

                if must_complete and not task_complete:
                    logging.critical("Required task not completed ... aborting!")
                    await render_model_output("🤖💥 *Required task not completed ...\n")
                    session.mark_failed(f"Required task {task_name!r} did not complete")
                    await render_model_output(
                        f"** 🤖💾 Session saved: {session.session_id}\n"
                        f"** 🤖💡 Resume with: --resume {session.session_id}\n"
                    )
                    break

                # Checkpoint after task (must_complete failures break above
                # without advancing the resume cursor)
                session.record_task(
                    index=task_index,
                    name=task_name,
                    success=task_complete,
                    tool_results=list(last_mcp_tool_results),
                )

        # All tasks completed successfully
        if session is not None and not session.error:
            session.mark_finished()
            await render_model_output(f"** 🤖✅ Session {session.session_id} completed\n")
