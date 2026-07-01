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
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jinja2

from ._stream import drive_backend_stream
from ._watchdog import start_watchdog, watchdog_ping
from .agent import DEFAULT_MODEL, TaskAgentHooks, TaskRunHooks
from .available_tools import AvailableTools
from .env_utils import TmpEnv
from .mcp_lifecycle import MCP_CLEANUP_TIMEOUT, build_mcp_servers, mcp_session_task
from .mcp_prompt import mcp_system_prompt
from .mcp_utils import compress_name, mcp_client_params
from .models import ModelConfigDocument, PersonalityDocument, TaskDefinition
from .render_utils import OutputRouter, flush_async_output, render_model_output, use_output_router
from .results import ResultStore, ToolResult, decode_tool_result, normalize_openai_tool_output
from .output_schema import validate_output
from .sdk import AgentSpec, MCPServerSpec, get_backend, resolve_backend_name
from .sdk.errors import (
    BackendBadRequestError,
    BackendMaxTurnsError,
    BackendTimeoutError,
    BackendUnexpectedError,
)
from .shell_utils import shell_tool_call
from .template_utils import evaluate_expression, render_template

if TYPE_CHECKING:  # Hook callbacks still use openai-agents types; fully decoupling them is a later slice.
    from agents import Agent, RunContextWrapper, TContext, Tool

DEFAULT_MAX_TURNS = 50  # Maximum agent turns before forced termination
RATE_LIMIT_BACKOFF = 5  # Initial backoff in seconds after a rate-limit response
MAX_RATE_LIMIT_BACKOFF = 120  # Maximum backoff cap in seconds for rate-limit retries
MAX_API_RETRY = 5  # Maximum number of consecutive API error retries
TASK_RETRY_LIMIT = 3  # Maximum retry attempts for a failed task
TASK_RETRY_BACKOFF = 10  # Initial backoff in seconds between task retries


def _resolve_model_config(
    available_tools: AvailableTools,
    model_config_ref: str,
) -> tuple[list[str], dict[str, str], dict[str, dict[str, Any]], str, str | None]:
    """Load and validate the model configuration file.

    Args:
        available_tools: Tool registry used to load the config file.
        model_config_ref: Reference name for the model config document.

    Returns:
        A tuple of (model_keys, model_dict, models_params, api_type, backend)
        where model_keys is the list of logical model names, model_dict maps
        them to provider model IDs, models_params holds per-model settings,
        api_type is ``"chat_completions"`` or ``"responses"``, and backend
        is the optional SDK adapter name from the config (``None`` when
        unset — the runner then falls back to env/endpoint resolution).

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
    return model_keys, model_dict, models_params, m_config.api_type, m_config.backend


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


def _resolve_one_model(
    logical_name: str,
    task_model_settings: dict[str, Any],
    model_keys: list[str],
    model_dict: dict[str, str],
    models_params: dict[str, dict[str, Any]],
    default_api_type: str = "chat_completions",
) -> tuple[str, dict[str, Any], str, str | None, str | None, str | None]:
    """Resolve one logical model name plus overrides into a concrete spec.

    Shared core for both the singular ``model:`` field and each entry of the
    multi-model ``models:`` list. *task_model_settings* are the per-task (or
    per-entry) overrides that win over the ``model_config`` settings.

    Returns:
        A tuple of ``(model_id, model_settings, api_type, endpoint, token, backend)``
        where *endpoint*, *token*, and *backend* are ``None`` when not overridden.
    """
    model_settings: dict[str, Any] = {}
    api_type: str = default_api_type
    endpoint: str | None = None
    token: str | None = None
    backend: str | None = None

    if logical_name in model_keys:
        if logical_name in models_params:
            model_settings = models_params[logical_name].copy()
        logical_name = model_dict[logical_name]

    # Extract engine-level keys before merging task settings
    api_type = model_settings.pop("api_type", api_type)
    endpoint = model_settings.pop("endpoint", None)
    token = model_settings.pop("token", None)
    backend = model_settings.pop("backend", None)

    # Task/entry-level overrides can also set engine keys
    task_settings = dict(task_model_settings)
    api_type = task_settings.pop("api_type", api_type)
    endpoint = task_settings.pop("endpoint", endpoint)
    token = task_settings.pop("token", token)
    backend = task_settings.pop("backend", backend)

    model_settings.update(task_settings)
    return logical_name, model_settings, api_type, endpoint, token, backend


def _resolve_task_model(
    task: TaskDefinition,
    model_keys: list[str],
    model_dict: dict[str, str],
    models_params: dict[str, dict[str, Any]],
    default_api_type: str = "chat_completions",
) -> tuple[str, dict[str, Any], str, str | None, str | None, str | None]:
    """Resolve the final model name, settings, and per-model overrides.

    Returns:
        A tuple of ``(model_id, model_settings, api_type, endpoint, token, backend)``
        where *endpoint*, *token*, and *backend* are ``None`` when not overridden.

    Raises:
        ValueError: If task-level model_settings is not a dictionary.
    """
    task_model_settings: dict[str, Any] | Any = task.model_settings or {}
    if not isinstance(task_model_settings, dict):
        raise ValueError(f"model_settings in task {task.name or ''} needs to be a dictionary")
    return _resolve_one_model(
        task.model or DEFAULT_MODEL,
        task_model_settings,
        model_keys,
        model_dict,
        models_params,
        default_api_type,
    )


@dataclass(frozen=True)
class ResolvedModel:
    """A fully resolved model spec for one branch of a (multi-)model task."""

    model: str
    model_settings: dict[str, Any]
    api_type: str
    endpoint: str | None
    token: str | None
    backend: str | None
    label: str


def _resolve_task_models(
    task: TaskDefinition,
    model_keys: list[str],
    model_dict: dict[str, str],
    models_params: dict[str, dict[str, Any]],
    default_api_type: str = "chat_completions",
) -> list[ResolvedModel]:
    """Resolve every model this task runs against.

    Single-model tasks yield a one-element list (equivalent to
    :func:`_resolve_task_model`); multi-model tasks yield one
    :class:`ResolvedModel` per ``models`` entry. The ``label`` is the
    user-facing logical name (falling back to the resolved provider id) and
    is used to tag per-model output streams.
    """
    resolved: list[ResolvedModel] = []
    for entry in task.effective_model_entries():
        entry_settings: dict[str, Any] | Any = entry.model_settings or {}
        if not isinstance(entry_settings, dict):
            raise ValueError(f"model_settings in task {task.name or ''} needs to be a dictionary")
        model_id, settings, api_type, endpoint, token, backend = _resolve_one_model(
            entry.model or DEFAULT_MODEL,
            entry_settings,
            model_keys,
            model_dict,
            models_params,
            default_api_type,
        )
        resolved.append(
            ResolvedModel(
                model=model_id,
                model_settings=settings,
                api_type=api_type,
                endpoint=endpoint,
                token=token,
                backend=backend,
                label=entry.model or model_id,
            )
        )
    return resolved


def _completion(oks: list[bool], policy: str) -> bool:
    """Reduce per-branch success flags to a single task result.

    An empty branch list is treated as success (nothing ran, nothing
    failed), matching the pre-multi-model behaviour. ``"any"`` succeeds if
    at least one branch succeeded; ``"all"`` (the default) requires every
    branch to succeed.
    """
    if not oks:
        return True
    if policy == "any":
        return any(oks)
    return all(oks)


def _capture_task_output(
    store: ResultStore,
    output_id: str,
    schema: dict[str, Any],
    task_name: str,
) -> None:
    """Capture a task's final tool result as a named typed output.

    The task's output is its most recent tool result. When *schema* is
    non-empty the decoded value is validated/coerced against it (raising a
    clear error on mismatch); otherwise the decoded value is stored as-is,
    falling back to the raw text when it is not JSON. The result is exposed to
    later tasks as ``outputs.<output_id>``.
    """
    last = store.last()
    if last is None:
        if schema:
            raise ValueError(
                f"task {task_name!r} declares 'outputs' but produced no tool result to capture"
            )
        store.set_output(output_id, None)
        return

    if schema:
        value = decode_tool_result(last)
        validated = validate_output(schema, value, model_name=f"{output_id}_output")
        store.set_output(output_id, validated)
    else:
        try:
            value = decode_tool_result(last)
        except ValueError:
            value = last.text
        store.set_output(output_id, value)


async def _fan_out_deploys(
    work_items: list[tuple[Any, ResolvedModel]],
    deploy: Callable[[Any, ResolvedModel], Awaitable[bool]],
    *,
    concurrent: bool,
    concurrency: int,
    completion_policy: str,
) -> bool:
    """Run ``deploy`` for every ``(payload, model)`` work item.

    This is the single place that owns the task fan-out matrix (prompts x
    models), the concurrency bound, exception isolation, and the completion
    policy, so the behaviour is unit-testable independently of the agent and
    MCP machinery.

    Args:
        work_items: ``(payload, resolved_model)`` pairs to execute. The
            payload is opaque to this helper and handed back to ``deploy``.
        deploy: coroutine factory invoked as ``deploy(payload, model)``,
            returning ``True`` on success.
        concurrent: when True all deploys run under a bounded ``gather`` and
            per-branch exceptions are captured and counted as failures; when
            False they run sequentially and exceptions propagate to the
            caller's retry loop (preserving the legacy single-model path).
        concurrency: maximum simultaneous deploys (clamped to >= 1) when
            ``concurrent`` is True.
        completion_policy: ``"all"`` or ``"any"`` (see :func:`_completion`).

    Returns:
        The reduced task success flag.
    """
    if not work_items:
        return True

    if not concurrent:
        # Sequential path: exceptions propagate exactly as the pre-multi-model
        # single-model path did, so the caller's retry loop still sees
        # transient backend errors.
        oks: list[bool] = []
        for payload, model in work_items:
            oks.append(await deploy(payload, model))
        return _completion(oks, completion_policy)

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _guarded(payload: Any, model: ResolvedModel) -> bool:
        async with semaphore:
            return await deploy(payload, model)

    gathered = await asyncio.gather(
        *(_guarded(payload, model) for payload, model in work_items),
        return_exceptions=True,
    )
    reduced: list[bool] = []
    for result in gathered:
        if not isinstance(result, bool):
            logging.error("Caught exception in Gather: %s", result, exc_info=result)
            result = False
        reduced.append(result)
    return _completion(reduced, completion_policy)


async def _build_prompts_to_run(
    task_prompt: str,
    repeat_prompt: bool,
    store: ResultStore,
    available_tools: AvailableTools,
    global_variables: dict[str, Any],
    inputs: dict[str, Any],
    outputs: dict[str, Any] | None = None,
    over: str = "",
) -> list[str]:
    """Build the list of prompts to execute for a task.

    For regular tasks the list contains a single rendered prompt.  When
    ``repeat_prompt`` is enabled, an iterable is derived and a prompt is
    rendered for each element:

    * If ``over`` is set, it is evaluated as an expression against the
      template context (``globals`` / ``inputs`` / ``outputs``) to yield the
      iterable directly. This is the explicit, typed path and does not consume
      the tool-result carry-over.
    * Otherwise the previous task's last tool result is decoded from *store*
      (the legacy path), and consumed (popped) after all prompts render.

    Args:
        task_prompt: The raw or pre-rendered prompt template string.
        repeat_prompt: Whether to expand prompts over an iterable.
        store: Per-run result store providing the last tool result.
        available_tools: Tool registry (passed through to template rendering).
        global_variables: Global template variables.
        inputs: Task-level input variables.
        outputs: Named task outputs available as ``outputs.<id>``.
        over: Optional expression selecting the iterable explicitly.

    Returns:
        List of rendered prompt strings to execute.

    Raises:
        IndexError: If the legacy path has no previous tool result.
        ValueError: If the derived value is not valid JSON or not iterable.
    """
    outputs = outputs or {}
    prompts_to_run: list[str] = []
    if repeat_prompt:
        if "result" not in task_prompt.lower():
            logging.warning("repeat_prompt enabled but no {{ result }} in prompt")

        if over:
            # Explicit, typed iterable: evaluate the expression against the
            # full template context. Does not consume the carry-over stack.
            try:
                iterable_result = evaluate_expression(
                    over,
                    available_tools,
                    globals_dict=global_variables,
                    inputs_dict=inputs,
                    outputs_dict=outputs,
                )
            except jinja2.TemplateError as exc:
                logging.critical("Could not evaluate over expression %r: %s", over, exc)
                raise ValueError(f"Failed to evaluate 'over' expression: {exc}") from exc
        else:
            last = store.last()
            if last is None:
                logging.critical("No last tool result available")
                raise IndexError("No last tool result available for repeat_prompt")
            iterable_result = decode_tool_result(last)

        try:
            iter(iterable_result)
        except TypeError:
            logging.critical("repeat_prompt iterable is not iterable: %r", iterable_result)
            raise

        if not iterable_result:
            await render_model_output("** 🤖❗repeat_prompt iterable is empty!\n")
        else:
            logging.debug("Rendering templated prompts for results: %s", iterable_result)
            for value in iterable_result:
                try:
                    rendered_prompt = render_template(
                        template_str=task_prompt,
                        available_tools=available_tools,
                        globals_dict=global_variables,
                        inputs_dict=inputs,
                        result_value=value,
                        outputs_dict=outputs,
                    )
                    prompts_to_run.append(rendered_prompt)
                except jinja2.TemplateError as e:
                    logging.error("Error rendering template for result %s: %s", value, e)
                    raise ValueError(f"Template rendering failed: {e}")

        # Legacy path consumes the tool result only after all prompts rendered
        # successfully, so it remains available for retry/resume on failure.
        # The explicit ``over`` path reads named data and never consumes.
        if not over:
            store.pop_last()
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
    backend: str | None = None,
    run_hooks: TaskRunHooks | None = None,
    agent_hooks: TaskAgentHooks | None = None,
    stream_label: str | None = None,
    record_tool_result: Callable[[ToolResult], Awaitable[None]] | None = None,
) -> bool:
    """Deploy and run task agents with MCP servers.

    Args:
        available_tools: Tool registry.
        agents: Mapping of agent name -> PersonalityDocument.
        prompt: User prompt to execute.
        api_type: OpenAI API type -- ``"chat_completions"`` or ``"responses"``.
        endpoint: Optional per-model API endpoint URL override.
        token: Optional env var name to resolve as the API token.
        backend: Optional explicit SDK adapter name (``"openai_agents"`` or
            ``"copilot_sdk"``). Defaults to ``SECLAB_TASKFLOW_BACKEND`` or
            an endpoint-based auto-default.
        stream_label: Optional human-readable label for this run's buffered
            output block (used to tag per-model streams in multi-model tasks).
            Only takes effect when ``async_task`` is True.
        record_tool_result: Neutral sink for tool results surfaced by backends
            that stream ``ToolEnd`` events (copilot/anthropic). The openai
            adapter captures results via ``run_hooks.on_tool_end`` instead.

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

    # Resolve the SDK adapter; it validates each spec just before build().
    backend_name = resolve_backend_name(explicit=backend, endpoint=endpoint)
    backend_impl = get_backend(backend_name)

    # Pass the user-provided model_settings through verbatim. The
    # openai-agents adapter applies its own tool_choice / temperature /
    # parallel_tool_calls defaults; the copilot adapter consumes only
    # ``reasoning_effort``.
    model_params: dict[str, Any] = dict(model_par)

    # Build MCP servers and collect server prompts
    entries = build_mcp_servers(available_tools, toolboxes, blocked_tools, headless)
    mcp_params = mcp_client_params(available_tools, toolboxes)
    server_prompts = [sp for _, (_, _, sp, _) in mcp_params.items()]

    # Wrap each built MCP server in a neutral spec. The openai
    # adapter unwraps ``params["_native"]``; the Copilot adapter reads
    # ``kind`` plus the raw transport keys.
    mcp_specs: list[MCPServerSpec] = []
    for entry in entries:
        raw_params, *_ = mcp_params.get(entry.server.name, ({}, [], "", 0.0))
        mcp_specs.append(
            MCPServerSpec(
                name=entry.server.name,
                kind=raw_params.get("kind", "stdio"),
                params={**raw_params, "_native": entry.server},
            )
        )

    # Connect MCP servers
    servers_connected = asyncio.Event()
    start_cleanup = asyncio.Event()
    mcp_sessions = asyncio.create_task(mcp_session_task(entries, servers_connected, start_cleanup))

    await servers_connected.wait()
    logging.debug("All mcp servers are connected!")

    agent_handle = None
    try:
        important_guidelines = [
            "Do not prompt the user with questions.",
            "Run tasks until a final result is available.",
            "Ensure responses are based on the latest information from available tools.",
            "Run tools sequentially, wait until one tool has completed before calling the next.",
        ]

        # Build handoff and primary AgentSpecs. The multi-personality case
        # sets in_handoff_graph=True on every participant so the openai
        # adapter can apply prompt_with_handoff_instructions equivalently
        # to the pre-refactor behaviour.
        agent_names = list(agents.keys())
        has_handoffs = len(agent_names) > 1
        handoff_specs: list[AgentSpec] = []
        for handoff_name in agent_names[1:]:
            personality = agents[handoff_name]
            handoff_specs.append(
                AgentSpec(
                    name=compress_name(handoff_name),
                    instructions=mcp_system_prompt(
                        personality.personality,
                        personality.task,
                        server_prompts=server_prompts,
                        important_guidelines=important_guidelines,
                    ),
                    model=model,
                    model_settings=model_params,
                    mcp_servers=mcp_specs,
                    handoffs=[],
                    exclude_from_context=exclude_from_context,
                    api_type=api_type,
                    endpoint=endpoint,
                    token_env=token,
                    in_handoff_graph=has_handoffs,
                    blocked_tools=blocked_tools,
                    headless=headless,
                )
            )

        primary_name = agent_names[0]
        primary_personality = agents[primary_name]
        primary_spec = AgentSpec(
            name=primary_name,
            instructions=mcp_system_prompt(
                primary_personality.personality,
                primary_personality.task,
                server_prompts=server_prompts,
                important_guidelines=important_guidelines,
            ),
            model=model,
            model_settings=model_params,
            mcp_servers=mcp_specs,
            handoffs=handoff_specs,
            exclude_from_context=exclude_from_context,
            api_type=api_type,
            endpoint=endpoint,
            token_env=token,
            in_handoff_graph=has_handoffs,
            blocked_tools=blocked_tools,
            headless=headless,
        )
        # Validate every spec (handoffs first, then primary) before
        # touching the backend so capability errors surface upfront.
        for spec in (*handoff_specs, primary_spec):
            backend_impl.validate(spec)
        agent_handle = await backend_impl.build(
            primary_spec, run_hooks=run_hooks, agent_hooks=agent_hooks
        )

        try:
            complete = False

            await drive_backend_stream(
                backend_impl=backend_impl,
                agent_handle=agent_handle,
                prompt=prompt,
                max_turns=max_turns,
                run_hooks=run_hooks,
                async_task=async_task,
                task_id=task_id,
                max_api_retry=MAX_API_RETRY,
                initial_rate_limit_backoff=RATE_LIMIT_BACKOFF,
                max_rate_limit_backoff=MAX_RATE_LIMIT_BACKOFF,
                record_tool_result=record_tool_result,
            )
            complete = True

        except BackendMaxTurnsError as e:
            await render_model_output(f"** 🤖❗ Max Turns Reached: {e}\n", async_task=async_task, task_id=task_id)
            logging.exception("Exceeded max_turns: %s", max_turns)
        except BackendUnexpectedError as e:
            await render_model_output(f"** 🤖❗ Agent Exception: {e}\n", async_task=async_task, task_id=task_id)
            logging.exception("Agent Exception")
        except BackendBadRequestError as e:
            await render_model_output(f"** 🤖❗ Request Error: {e}\n", async_task=async_task, task_id=task_id)
            logging.exception("Bad Request")
        except BackendTimeoutError as e:
            await render_model_output(f"** 🤖❗ Timeout Error: {e}\n", async_task=async_task, task_id=task_id)
            logging.exception("API Timeout")

        if async_task:
            await flush_async_output(task_id, label=stream_label)

        return complete

    finally:
        watchdog_ping()
        if agent_handle is not None:
            try:
                await backend_impl.aclose(agent_handle)
            except Exception:  # noqa: BLE001 - best-effort release
                logging.exception("Backend aclose failed")
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
        # Yield to give mcp_session_task a chance to finish after being
        # signalled, especially when there are no servers to clean up.
        await asyncio.sleep(0)
        # If the MCP session task is still running (e.g. all servers were
        # already disconnected and the cleanup loop above never entered)
        # cancel it explicitly so a dangling task can't keep the event
        # loop alive past run_main.
        if not mcp_sessions.done():
            mcp_sessions.cancel()
            try:
                await asyncio.wait_for(mcp_sessions, timeout=MCP_CLEANUP_TIMEOUT)
            except asyncio.TimeoutError:
                logging.warning(
                    "Timed out waiting for MCP session task cancellation after %ds",
                    MCP_CLEANUP_TIMEOUT,
                )
            except asyncio.CancelledError:
                pass
            except Exception:
                logging.exception("Exception while cancelling MCP session task")


async def run_main(
    available_tools: AvailableTools,
    personality_path: str | None,
    taskflow_path: str | None,
    cli_globals: dict[str, str],
    prompt: str | None,
    resume_session_id: str | None = None,
    cli_model_config: str | None = None,
) -> None:
    """Main entry point for taskflow/personality execution.

    Args:
        available_tools: Tool registry.
        personality_path: Personality module path, or None.
        taskflow_path: Taskflow module path, or None.
        cli_globals: Global variables from CLI.
        prompt: User prompt text.
        resume_session_id: Session ID to resume from a checkpoint.
        cli_model_config: Model configuration module path, or None.
    """
    from .session import TaskflowSession

    # Daemon thread that force-exits the process if the event loop stops
    # making progress for any reason the asyncio-layer timeouts didn't
    # already handle. Idempotent — safe to call on every run_main.
    start_watchdog()

    # Give this run its own output router so buffered async / multi-model
    # streams stay isolated from any other run sharing the process.
    use_output_router(OutputRouter())

    store = ResultStore()

    async def on_tool_end_hook(context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str) -> None:
        # openai-agents delivers tool results here as a backend-specific
        # serialised string; normalise to a neutral ToolResult before storing.
        watchdog_ping()
        store.record(normalize_openai_tool_output(result, tool_name=getattr(tool, "name", "")))

    async def record_tool_result(tool_result: ToolResult) -> None:
        # Neutral sink for backends (copilot/anthropic) that surface tool
        # results as stream events rather than via openai-agents RunHooks.
        watchdog_ping()
        store.record(tool_result)

    async def on_tool_start_hook(context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool) -> None:
        watchdog_ping()
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
            record_tool_result=record_tool_result,
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
            store = ResultStore.from_snapshot(session.result_snapshot)
            # Restore persisted model config unless explicitly overridden
            if not cli_model_config and session.cli_model_config:
                cli_model_config = session.cli_model_config
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
        if cli_model_config:
            model_config_ref = cli_model_config
        model_keys: list[str] = []
        model_dict: dict[str, str] = {}
        models_params: dict[str, dict[str, Any]] = {}
        api_type: str = "chat_completions"
        backend: str | None = None
        if model_config_ref:
            model_keys, model_dict, models_params, api_type, backend = _resolve_model_config(
                available_tools, model_config_ref
            )

        # Create session if this is a new run (not personality mode)
        if session is None:
            session = TaskflowSession(
                taskflow_path=taskflow_path,
                cli_globals=cli_globals,
                prompt=prompt or "",
                total_tasks=len(taskflow_doc.taskflow),
                cli_model_config=cli_model_config or "",
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

            # Resolve models (one per model entry; single-model tasks yield
            # a one-element list). Multi-model tasks fan out over this list.
            resolved_models = _resolve_task_models(
                task, model_keys, model_dict, models_params, default_api_type=api_type,
            )
            multi_model = len(resolved_models) > 1

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
            completion_policy = task.completion
            # Bound on concurrent model runs (0 == run all models at once).
            model_concurrency = task.model_concurrency or len(resolved_models)
            # Typed named outputs (M2): id names the task's captured output;
            # outputs declares its schema; over selects a repeat_prompt iterable.
            task_output_id = task.id
            task_output_schema = task.outputs or {}
            over = task.over or ""

            # Render prompt template (skip if repeat_prompt — result not yet available)
            if task_prompt and not repeat_prompt:
                try:
                    task_prompt = render_template(
                        template_str=task_prompt,
                        available_tools=available_tools,
                        globals_dict=global_variables,
                        inputs_dict=inputs,
                        outputs_dict=store.outputs,
                    )
                except jinja2.TemplateError as e:
                    logging.error("Template rendering error: %s", e)
                    raise ValueError(f"Failed to render prompt template: {e}") from e

            with TmpEnv(env, context={"globals": global_variables}):
                prompts_to_run: list[str] = await _build_prompts_to_run(
                    task_prompt, repeat_prompt, store,
                    available_tools, global_variables, inputs,
                    outputs=store.outputs, over=over,
                )

                async def run_prompts(async_task: bool = False, max_concurrent_tasks: int = 5) -> bool:
                    if run:
                        await render_model_output("** 🤖🐚 Executing Shell Task\n")
                        try:
                            content = shell_tool_call(run).content[0]
                            store.record(ToolResult(tool_name="shell", text=getattr(content, "text", "")))
                            return True
                        except RuntimeError as e:
                            await render_model_output(f"** 🤖❗ Shell Task Exception: {e}\n")
                            logging.exception("Shell task error")
                            return False

                    # Concurrency: repeat_prompt async fans out over prompts;
                    # multi-model fans out over models. Either triggers the
                    # buffered/gathered path so per-branch output stays intact.
                    concurrent = async_task or multi_model
                    concurrency = model_concurrency if multi_model else max_concurrent_tasks

                    # Resolve agents (and rewrite bare prompts) once per prompt,
                    # before fanning out across models so the work is not
                    # repeated per model.
                    resolved_prompts: list[tuple[dict[str, Any], str]] = []
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
                        resolved_prompts.append((resolved_agents, p_prompt))

                    async def _deploy(payload: tuple[dict[str, Any], str], rm: ResolvedModel) -> bool:
                        ra, pp = payload
                        # Isolate multi-model tool-result capture: concurrent
                        # models must not interleave into the shared result
                        # store (which feeds downstream repeat_prompt / typed
                        # outputs / session checkpoints). Single-model runs keep
                        # the shared sink for backwards compatibility.
                        if multi_model:
                            async def _isolated_tool_end(context, agent, tool, result) -> None:  # noqa: ANN001
                                watchdog_ping()
                            run_hooks = TaskRunHooks(
                                on_tool_end=_isolated_tool_end, on_tool_start=on_tool_start_hook
                            )
                            branch_record = None
                        else:
                            run_hooks = TaskRunHooks(
                                on_tool_end=on_tool_end_hook, on_tool_start=on_tool_start_hook
                            )
                            branch_record = record_tool_result
                        return await deploy_task_agents(
                            available_tools,
                            ra,
                            pp,
                            async_task=concurrent,
                            toolboxes_override=toolboxes_override,
                            blocked_tools=blocked_tools,
                            headless=headless,
                            exclude_from_context=exclude_from_context,
                            max_turns=max_turns,
                            run_hooks=run_hooks,
                            record_tool_result=branch_record,
                            model=rm.model,
                            model_par=rm.model_settings,
                            api_type=rm.api_type,
                            endpoint=rm.endpoint,
                            token=rm.token,
                            backend=rm.backend or backend,
                            agent_hooks=TaskAgentHooks(on_handoff=on_handoff_hook),
                            stream_label=rm.label if multi_model else None,
                        )

                    # Fan-out matrix: every (prompt, model) pair.
                    work_items = [
                        (payload, rm)
                        for payload in resolved_prompts
                        for rm in resolved_models
                    ]
                    return await _fan_out_deploys(
                        work_items,
                        _deploy,
                        concurrent=concurrent,
                        concurrency=concurrency,
                        completion_policy=completion_policy,
                    )

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
                    except (BackendTimeoutError, ConnectionError, TimeoutError) as exc:
                        last_task_error = exc
                        remaining = TASK_RETRY_LIMIT - attempt - 1
                        if remaining > 0:
                            backoff = TASK_RETRY_BACKOFF * (attempt + 1)
                            await render_model_output(
                                f"** 🤖🔄 Task {task_name!r} failed: {exc}\n"
                                f"** 🤖🔄 Retrying in {backoff}s ({remaining} attempts left)\n"
                            )
                            logging.warning("Task %r attempt %s failed: %s", task_name, attempt + 1, exc)
                            await asyncio.sleep(backoff)
                        else:
                            logging.error(
                                "Task %r failed after %s attempts: %s", task_name, TASK_RETRY_LIMIT, exc
                            )
                    except Exception as exc:
                        last_task_error = exc
                        logging.error("Task %r failed (non-retriable): %s", task_name, exc)
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

                # Capture this task's typed named output (M2). The task's
                # output is its final tool result; when an ``outputs`` schema
                # is declared it is validated/coerced before being stored under
                # ``outputs.<id>`` for downstream tasks to consume by name.
                if task_output_id and not multi_model:
                    _capture_task_output(
                        store, task_output_id, task_output_schema, task_name
                    )

                # Checkpoint after task (must_complete failures break above
                # without advancing the resume cursor)
                session.record_task(
                    index=task_index,
                    name=task_name,
                    success=task_complete,
                    result_snapshot=store.snapshot(),
                )

        # All tasks completed successfully
        if session is not None and not session.error:
            session.mark_finished()
            await render_model_output(f"** 🤖✅ Session {session.session_id} completed\n")
