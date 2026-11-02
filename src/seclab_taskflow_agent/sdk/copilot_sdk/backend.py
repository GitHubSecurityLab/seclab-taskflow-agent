# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""GitHub Copilot SDK (1.x) backend adapter.

* :meth:`build` starts a :class:`copilot.CopilotClient` authenticated with
  the token from ``spec.token_env`` (default ``COPILOT_GITHUB_TOKEN``),
  verifies the requested model is available, creates a session bound to
  that model and the agent's policy-enforcing MCP tools, and wires a queue that captures
  every ``SessionEvent``.
* :meth:`run_streamed` calls ``session.send`` and drains the queue,
  yielding neutral stream events until the ``session.idle`` event for the
  sent message (success) or a ``session.error`` (translated to a neutral
  backend error). Any model switch away from the requested model is an
  error, never a silent fallback.
* :meth:`aclose` disconnects the session and stops the client.
"""

from __future__ import annotations

__all__ = ["CopilotSDKBackend"]

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..base import AgentSpec, StreamEvent, TextDelta, TokenUsage, ToolEnd
from ..errors import (
    BackendBadRequestError,
    BackendCapabilityError,
    BackendError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnexpectedError,
)
from .mcp import build_mcp_tools
from .permissions import build_permission_handler

if TYPE_CHECKING:
    from copilot.session import ReasoningEffort

_VALID_REASONING = ("low", "medium", "high", "xhigh", "max")

#: Env var consulted when the taskflow does not name one via ``token:``.
DEFAULT_TOKEN_ENV = "COPILOT_GITHUB_TOKEN"  # noqa: S105 - environment variable name

#: Copilot's router alias. Requesting it explicitly opts out of model pinning.
AUTO_MODEL = "auto"

_BAD_REQUEST_STATUS = frozenset({400, 401, 403, 404, 422})
_TIMEOUT_STATUS = frozenset({408, 504})


@dataclass
class _CopilotHandle:
    client: Any
    session: Any
    event_queue: asyncio.Queue[Any]
    exclude_from_context: bool = False
    model: str = ""


def _resolve_token(token_env: str | None) -> str | None:
    """Resolve the GitHub token from an environment variable *name*.

    * ``token_env`` set: the variable must be non-empty, otherwise
      :class:`BackendBadRequestError` is raised (fail closed rather than
      silently using whatever identity the Copilot CLI is logged in as).
    * ``token_env`` unset: :data:`DEFAULT_TOKEN_ENV` is used when present;
      otherwise ``None`` is returned and the SDK's default auth chain
      applies.

    Never returns the env var *name* itself and never includes the token
    value in error messages.
    """
    if token_env:
        value = os.getenv(token_env)
        if not value:
            raise BackendBadRequestError(f"copilot_sdk: token environment variable {token_env!r} is unset or empty")
        return value
    return os.getenv(DEFAULT_TOKEN_ENV) or None


def _normalize_model(model: str) -> str:
    """Strip the provider prefix (``openai/gpt-4.1`` → ``gpt-4.1``).

    The Copilot SDK selects models by bare name; the provider-prefixed
    form is a CAPI inference convention used by the openai-agents path.

    Raises:
        BackendBadRequestError: If *model* is empty. The Copilot SDK
            silently falls back to its built-in default model when no
            ``model`` is passed to ``create_session``, which would
            invalidate any reproducibility guarantee a taskflow makes
            about the model under test.
    """
    model = model.split("/", 1)[-1].strip()
    if not model:
        raise BackendBadRequestError("copilot_sdk: model is required (the SDK would otherwise pick a default)")
    return model


def _tool_result_text(data: Any) -> str | None:
    """Extract the textual payload from a ``tool.execution_complete`` event.

    The Copilot SDK delivers MCP tool results as a ``Result`` object with a
    list of content entries. We concatenate the ``text`` fields of the text
    entries, mirroring how openai-agents stringifies an MCP text content
    list before handing it to the runner's tool-end hook.
    """
    result = getattr(data, "result", None)
    if result is None:
        return None
    contents = getattr(result, "contents", None) or []
    parts = [getattr(c, "text", "") for c in contents if getattr(c, "text", None)]
    if parts:
        return "".join(parts)
    content = getattr(result, "content", None)
    return content if isinstance(content, str) else None


def _reasoning_effort(model_settings: dict[str, Any]) -> ReasoningEffort | None:
    raw = model_settings.get("reasoning_effort")
    if raw is None:
        return None
    if raw not in _VALID_REASONING:
        raise BackendBadRequestError(
            f"copilot_sdk: invalid reasoning_effort {raw!r} (expected one of {_VALID_REASONING})"
        )
    return raw


def _translate_exception(exc: Exception, action: str) -> Exception:
    """Map an exception raised by an SDK call to a neutral backend error."""
    if isinstance(exc, BackendError):
        return exc
    if isinstance(exc, TimeoutError):
        return BackendTimeoutError(f"copilot_sdk: {action} timed out")
    return BackendUnexpectedError(f"copilot_sdk: {action} failed: {exc}")


def _session_error(data: Any) -> Exception:
    """Translate ``session.error`` event data into a neutral backend error."""
    message = getattr(data, "message", None) or "session error"
    error_type = str(getattr(data, "error_type", "") or "").lower()
    status = getattr(data, "status_code", None)
    detail = f"copilot_sdk: {message}"
    if error_type:
        detail += f" (error_type={error_type}"
        detail += f", status={status})" if status is not None else ")"
    if error_type == "rate_limit" or status == 429:
        return BackendRateLimitError(detail)
    if error_type == "timeout" or status in _TIMEOUT_STATUS:
        return BackendTimeoutError(detail)
    if error_type in ("authentication", "authorization") or status in _BAD_REQUEST_STATUS:
        return BackendBadRequestError(detail)
    return BackendUnexpectedError(detail)


def _model_mismatch(expected: str, actual: str | None) -> bool:
    return bool(actual) and actual != expected


def _check_model(event: Any, expected: str) -> None:
    """Raise if *event* shows the session running a model other than *expected*."""
    from copilot.session_events import SessionEventType

    etype = getattr(event, "type", None)
    data = getattr(event, "data", None)
    if etype == SessionEventType.SESSION_MODEL_CHANGE:
        actual = getattr(data, "new_model", None)
        source = "session.model_change"
    elif etype == SessionEventType.SESSION_START:
        actual = getattr(data, "selected_model", None)
        source = "session.start"
    elif etype in (SessionEventType.ASSISTANT_MESSAGE, SessionEventType.ASSISTANT_USAGE):
        actual = getattr(data, "model", None)
        source = str(etype.value)
    else:
        return
    if _model_mismatch(expected, actual):
        raise BackendUnexpectedError(
            f"copilot_sdk: requested model {expected!r} but {source} reports "
            f"{actual!r}; refusing to continue with a different model"
        )


async def _verify_model(client: Any, model: str) -> None:
    """Validate concrete models; ``auto`` is a router alias, not a catalogue ID."""
    if model == AUTO_MODEL:
        return
    models = await client.list_models()
    for info in models:
        if getattr(info, "id", None) != model:
            continue
        policy = getattr(info, "policy", None)
        if getattr(policy, "state", None) == "disabled":
            raise BackendBadRequestError(f"copilot_sdk: model {model!r} is disabled by policy for this account")
        return
    raise BackendBadRequestError(
        f"copilot_sdk: model {model!r} is not available to this account ({len(models)} models listed)"
    )


async def _stop_client(client: Any) -> None:
    try:
        await client.stop()
    except Exception:
        logging.exception("copilot_sdk: client.stop failed")


class CopilotSDKBackend:
    """Adapter that drives the GitHub Copilot SDK."""

    name = "copilot_sdk"

    def validate(self, spec: AgentSpec) -> None:
        """Reject YAML fields the Copilot SDK cannot honour.

        Handoffs:
            The openai-agents handoff model is "the LLM picks the next
            agent via a tool call, the SDK swaps instructions+tools+
            model in place." The Copilot SDK lets you preregister
            personas via ``custom_agents`` and pick one at session
            start, but exposes no API to switch the active persona
            mid-session — ``system_message`` is only accepted by
            ``create_session`` and ``set_model`` only swaps the model.
            We could fake it by destroying and recreating the session
            with replayed history, but that breaks tool-call state and
            depends on undocumented behaviour. Reject up front instead.

        Model settings:
            ``temperature`` and ``parallel_tool_calls`` have no analogue
            in ``CopilotClient.create_session``; silently dropping them
            would mask configuration drift.

        ``api_type`` is intentionally ignored — Copilot picks its own
        wire protocol per model.
        """
        if spec.handoffs or spec.in_handoff_graph:
            raise BackendCapabilityError("copilot_sdk: agent handoffs are not supported")
        for unsupported in ("temperature", "parallel_tool_calls"):
            if unsupported in spec.model_settings:
                raise BackendCapabilityError(f"copilot_sdk: model_settings.{unsupported} is not supported")

    async def build(
        self,
        spec: AgentSpec,
        *,
        run_hooks: Any = None,
        agent_hooks: Any = None,
    ) -> _CopilotHandle:
        del run_hooks, agent_hooks  # no parity yet
        from copilot import CopilotClient, ToolSet

        model = _normalize_model(spec.model)
        reasoning_effort = _reasoning_effort(spec.model_settings)
        token = _resolve_token(spec.token_env)

        client = CopilotClient(github_token=token, use_logged_in_user=False) if token else CopilotClient()

        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[Any] = asyncio.Queue()

        def _on_event(event: Any) -> None:
            with contextlib.suppress(RuntimeError):
                # Loop may have closed during shutdown.
                loop.call_soon_threadsafe(event_queue.put_nowait, event)

        try:
            await client.start()
            auth = await client.get_auth_status()
            if not auth.isAuthenticated:
                raise BackendBadRequestError(
                    "copilot_sdk: authentication failed; provide a Copilot-enabled "
                    "GitHub token or sign in with the Copilot CLI"
                )
            await _verify_model(client, model)
            tools = await build_mcp_tools(
                spec.mcp_servers,
                blocked_tools=spec.blocked_tools,
                exclude_from_context=spec.exclude_from_context,
            )
            available_tools = ToolSet()
            for tool in tools:
                available_tools.add_custom(tool.name)
            session = await client.create_session(
                model=model,
                on_permission_request=build_permission_handler([tool.name for tool in tools]),
                tools=tools,
                available_tools=available_tools,
                # The custom-only policy cannot retrieve SDK-offloaded files.
                large_output={"enabled": False},
                streaming=True,
                include_sub_agent_streaming_events=False,
                reasoning_effort=reasoning_effort,
                system_message=({"mode": "replace", "content": spec.instructions} if spec.instructions else None),
                # Preserve SDK authentication defaults without discovering
                # ambient instructions, hooks, tools or cross-session memory.
                enable_config_discovery=False,
                skip_custom_instructions=True,
                enable_on_demand_instruction_discovery=False,
                enable_file_hooks=False,
                enable_host_git_operations=False,
                enable_session_store=False,
                enable_skills=False,
                skip_embedding_retrieval=True,
                memory={"enabled": False},
                on_event=_on_event,
            )
        except BaseException as exc:
            await _stop_client(client)
            translated = _translate_exception(exc, "session setup") if isinstance(exc, Exception) else exc
            if translated is exc:
                raise
            raise translated from exc

        return _CopilotHandle(
            client=client,
            session=session,
            event_queue=event_queue,
            exclude_from_context=spec.exclude_from_context,
            model=model,
        )

    async def run_streamed(
        self,
        agent: _CopilotHandle,
        prompt: str,
        *,
        max_turns: int,
    ) -> AsyncIterator[StreamEvent]:
        del max_turns  # the SDK manages turn limits internally
        from copilot.session_events import SessionEventType

        expected_model = getattr(agent, "model", "") or ""
        routed_models: set[str] = set()

        def _enforce_model(event: Any) -> None:
            if not expected_model:
                return
            if expected_model == AUTO_MODEL:
                # The taskflow explicitly asked Copilot to route; record
                # which concrete model served it instead of pinning.
                actual = getattr(getattr(event, "data", None), "model", None)
                if actual and actual not in routed_models:
                    routed_models.add(actual)
                    logging.info("copilot_sdk: model 'auto' routed to %r", actual)
                return
            _check_model(event, expected_model)

        started = False
        completed = False
        tool_names: dict[str, str] = {}
        streamed_message_ids: set[str] = set()
        try:
            # Only session-level events describe the model for the next request.
            while True:
                try:
                    stale = agent.event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if getattr(stale, "type", None) in (
                    SessionEventType.SESSION_START,
                    SessionEventType.SESSION_MODEL_CHANGE,
                ) and not getattr(stale, "agent_id", None):
                    _enforce_model(stale)

            try:
                message_id = await agent.session.send(prompt)
            except Exception as exc:
                translated = _translate_exception(exc, "session.send")
                if translated is exc:
                    raise
                raise translated from exc

            while True:
                event = await agent.event_queue.get()
                etype = getattr(event, "type", None)
                data = getattr(event, "data", None)

                if etype == SessionEventType.ASSISTANT_MESSAGE and getattr(
                    data, "originating_message_id", None
                ) not in (None, message_id):
                    continue

                if getattr(event, "agent_id", None):
                    # Sub-agent events: account for usage only.
                    if etype == SessionEventType.ASSISTANT_USAGE:
                        yield _token_usage(data)
                    continue

                _enforce_model(event)

                if (
                    etype == SessionEventType.SESSION_WARNING
                    and getattr(data, "warning_type", None) == "policy_blocked"
                    and getattr(data, "originating_message_id", None) == message_id
                ):
                    raise BackendBadRequestError(
                        f"copilot_sdk: {getattr(data, 'message', None) or 'Request blocked by policy'}"
                    )

                if not started:
                    if etype == SessionEventType.USER_MESSAGE:
                        started = getattr(data, "message_id", None) == message_id
                        continue
                    if etype == SessionEventType.ASSISTANT_MESSAGE and (
                        getattr(data, "originating_message_id", None) == message_id
                    ):
                        started = True
                    elif etype == SessionEventType.SESSION_ERROR and getattr(data, "originating_message_id", None) in (
                        None,
                        message_id,
                    ):
                        raise _session_error(data)
                    else:
                        continue

                if etype == SessionEventType.ASSISTANT_MESSAGE_DELTA:
                    if getattr(data, "parent_tool_call_id", None):
                        continue
                    text = getattr(data, "delta_content", None) or ""
                    if text:
                        streamed_message_ids.add(getattr(data, "message_id", None) or "")
                        yield TextDelta(text=text)
                elif etype == SessionEventType.ASSISTANT_MESSAGE:
                    # Fallback for messages that were not streamed as deltas.
                    if getattr(data, "parent_tool_call_id", None):
                        continue
                    if (getattr(data, "message_id", None) or "") not in streamed_message_ids:
                        content = getattr(data, "content", None) or ""
                        if content:
                            yield TextDelta(text=content)
                elif etype == SessionEventType.ASSISTANT_USAGE:
                    yield _token_usage(data)
                elif etype == SessionEventType.TOOL_EXECUTION_START:
                    call_id = getattr(data, "tool_call_id", None)
                    if call_id:
                        tool_names[call_id] = (
                            getattr(data, "mcp_tool_name", None) or getattr(data, "tool_name", None) or ""
                        )
                elif etype == SessionEventType.TOOL_EXECUTION_COMPLETE:
                    call_id = getattr(data, "tool_call_id", None)
                    tool_name = tool_names.pop(call_id, "") if call_id else ""
                    if not getattr(data, "success", False):
                        error = getattr(data, "error", None)
                        logging.debug(
                            "copilot_sdk: tool %r failed: %s",
                            tool_name,
                            getattr(error, "message", None) or "unknown error",
                        )
                        continue
                    text = _tool_result_text(data)
                    if text is not None:
                        yield ToolEnd(tool_name=tool_name, text=text)
                    # The registered terminal tool prevents a follow-up
                    # model call; abort also releases any remaining work.
                    if agent.exclude_from_context:
                        with contextlib.suppress(Exception):
                            await agent.session.abort()
                        completed = True
                        return
                elif etype == SessionEventType.SESSION_ERROR:
                    if getattr(data, "originating_message_id", None) in (None, message_id):
                        raise _session_error(data)
                elif etype == SessionEventType.SESSION_IDLE:
                    mode = getattr(data, "mode", None)
                    if getattr(mode, "value", mode) == "autopilot":
                        continue
                    if getattr(data, "aborted", False):
                        raise BackendUnexpectedError("copilot_sdk: session aborted before the response completed")
                    completed = True
                    return
        finally:
            # If the consumer abandoned the iterator (idle-timeout, error,
            # caller break) the session may still be mid-turn. Abort so the
            # runtime stops generating and aclose() can run cleanly.
            if not completed:
                with contextlib.suppress(Exception):
                    await agent.session.abort()

    async def aclose(self, agent: _CopilotHandle | None) -> None:
        if agent is None:
            return
        try:
            try:
                await agent.session.disconnect()
            except Exception:
                logging.exception("copilot_sdk: session.disconnect failed")
        finally:
            await _stop_client(agent.client)


def _token_usage(data: Any) -> TokenUsage:
    # Copilot reports usage from the chat_completions surface, so (like
    # OpenAI) ``input_tokens`` INCLUDES cached tokens and
    # ``cache_read_tokens`` is a subset already counted in it.
    return TokenUsage(
        model=getattr(data, "model", "") or "",
        input_tokens=int(getattr(data, "input_tokens", 0) or 0),
        output_tokens=int(getattr(data, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(data, "cache_read_tokens", 0) or 0),
        cache_write_tokens=int(getattr(data, "cache_write_tokens", 0) or 0),
    )
