# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Stream-driving helpers for the runner.

This module owns the inner loop that consumes events from a backend
adapter (`TextDelta` / `ToolEnd`), renders text deltas to the user, and
bridges Copilot-side tool events into the run-hook callbacks that the
runner uses to capture MCP results for ``repeat_prompt`` and session
checkpointing.

Extracted from ``runner.py`` so the rate-limit/retry loop and the
backend-event translation are independently readable and testable.
"""

from __future__ import annotations

__all__ = ["STREAM_IDLE_TIMEOUT", "handle_tool_end_event", "drive_backend_stream"]

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

from ._watchdog import watchdog_ping
from .render_utils import render_model_output
from .results import ToolResult
from .sdk import TextDelta, TokenUsage, ToolEnd
from .sdk.errors import BackendRateLimitError, BackendTimeoutError

# Application-level backstop: if the backend's event stream goes silent
# for this long, surface a BackendTimeoutError so the retry loop can
# recover. This complements the TCP-level httpx timeouts in the
# openai-agents adapter — those catch dead sockets, this catches the
# subtler case where the connection stays open but nothing is flowing.
STREAM_IDLE_TIMEOUT = 1800


async def handle_tool_end_event(
    event: ToolEnd,
    run_hooks: Any,
    record_tool_result: Any = None,
) -> None:
    """Handle a backend ``ToolEnd`` stream event (copilot/anthropic).

    Renders the tool-call progress notice via ``run_hooks.on_tool_start``
    (the openai path renders this natively) and forwards a neutral
    :class:`ToolResult` to *record_tool_result* so the runner captures it in
    its result store.

    Unlike the previous implementation, this does **not** reconstruct an
    openai-agents-specific ``{"text": ...}`` JSON envelope: the runner now
    consumes a neutral :class:`ToolResult` directly, so no backend has to fake
    another backend's wire format.
    """
    fake_tool = SimpleNamespace(name=event.tool_name)
    if run_hooks is not None:
        await run_hooks.on_tool_start(None, None, fake_tool)
    if record_tool_result is not None:
        await record_tool_result(ToolResult(tool_name=event.tool_name, text=event.text))


async def drive_backend_stream(
    *,
    backend_impl: Any,
    agent_handle: Any,
    prompt: str,
    max_turns: int,
    run_hooks: Any,
    async_task: bool,
    task_id: str,
    max_api_retry: int,
    initial_rate_limit_backoff: int,
    max_rate_limit_backoff: int,
    record_tool_result: Any = None,
    record_usage: Any = None,
    record_message: Any = None,
) -> None:
    """Run the backend's event stream to completion with retry/backoff.

    Renders ``TextDelta`` events to stdout, forwards ``ToolEnd`` events to
    :func:`handle_tool_end_event` (which records a neutral
    :class:`~seclab_taskflow_agent.results.ToolResult` via
    *record_tool_result*), logs and forwards ``TokenUsage`` events to
    *record_usage*, retries up to *max_api_retry* times on
    :class:`BackendTimeoutError`, and applies exponential backoff up to
    *max_rate_limit_backoff* seconds on :class:`BackendRateLimitError`
    before giving up with a :class:`BackendTimeoutError`.

    When *record_message* is provided, the agent's final response text is
    forwarded to it once the stream completes successfully. "Final response"
    is the prose emitted after the last tool call: text deltas are accumulated
    and reset on each ``ToolEnd``, so a task that ends on a tool call yields
    only that trailing summary, and a plain question-and-answer turn yields the
    whole answer. This is what ``capture: response`` tasks store as their named
    output.
    """
    max_retry = max_api_retry
    rate_limit_backoff = initial_rate_limit_backoff
    last_rate_limit_exc: BackendRateLimitError | None = None

    while rate_limit_backoff:
        # Accumulate the agent's final response text per attempt (reset on a
        # retry, and on each tool call, so we keep only the prose after the
        # last tool result -- the model's final answer).
        final_message_parts: list[str] = []
        try:
            stream = backend_impl.run_streamed(
                agent_handle, prompt, max_turns=max_turns
            )
            stream_iter = stream.__aiter__()
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(
                            stream_iter.__anext__(), timeout=STREAM_IDLE_TIMEOUT
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError as exc:
                        raise BackendTimeoutError(
                            f"Backend stream idle for {STREAM_IDLE_TIMEOUT}s"
                        ) from exc
                    watchdog_ping()
                    if isinstance(event, TextDelta):
                        final_message_parts.append(event.text)
                        await render_model_output(
                            event.text, async_task=async_task, task_id=task_id
                        )
                    elif isinstance(event, ToolEnd):
                        final_message_parts.clear()
                        await handle_tool_end_event(event, run_hooks, record_tool_result)
                    elif isinstance(event, TokenUsage):
                        logging.info(
                            "token usage model=%s input=%d output=%d "
                            "cache_write=%d cache_read=%d",
                            event.model,
                            event.input_tokens,
                            event.output_tokens,
                            event.cache_write_tokens,
                            event.cache_read_tokens,
                        )
                        if record_usage is not None:
                            await record_usage(event)
            finally:
                # Close the async generator so its finally block runs even
                # if we abort early (timeout / consumer break) — the
                # adapters use that to release backend-native resources.
                aclose = getattr(stream_iter, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:  # noqa: BLE001 - best-effort cleanup
                        logging.exception("Failed to aclose backend stream iterator")
            await render_model_output("\n\n", async_task=async_task, task_id=task_id)
            if record_message is not None:
                await record_message("".join(final_message_parts))
            return
        except BackendTimeoutError:
            if not max_retry:
                logging.exception("Max retries for BackendTimeoutError reached")
                raise
            max_retry -= 1
        except BackendRateLimitError as exc:
            last_rate_limit_exc = exc
            if rate_limit_backoff == max_rate_limit_backoff:
                raise BackendTimeoutError("Max rate limit backoff reached") from exc
            if rate_limit_backoff > max_rate_limit_backoff:
                rate_limit_backoff = max_rate_limit_backoff
            else:
                rate_limit_backoff += rate_limit_backoff
            logging.exception("Hit rate limit ... holding for %s", rate_limit_backoff)
            await asyncio.sleep(rate_limit_backoff)

    if last_rate_limit_exc is not None:  # pragma: no cover - loop always returns/raises above
        raise BackendTimeoutError("Rate limit backoff exhausted") from last_rate_limit_exc
