# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Anthropic SDK backend adapter.

Drives the Anthropic Messages API (``/v1/messages``) via the official
``anthropic`` Python SDK. Supports streaming, tool calling via MCP
servers, and extended thinking.

Auth note: The Anthropic SDK sends ``x-api-key`` by default, but
providers that use Bearer auth (see ``APIProvider.bearer_auth``)
need ``Authorization: Bearer`` instead.  We pass the bearer header
via ``default_headers`` and set ``api_key`` to a placeholder so the
SDK doesn't send the real token via x-api-key.
"""

from __future__ import annotations

__all__ = ["AnthropicSDKBackend"]

import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..base import AgentSpec, StreamEvent, TextDelta, TokenUsage, ToolEnd
from ..errors import (
    BackendBadRequestError,
    BackendCapabilityError,
    BackendMaxTurnsError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnexpectedError,
)

logger = logging.getLogger(__name__)

_VALID_REASONING = ("low", "medium", "high", "max")


def _mcp_tools_to_anthropic(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to Anthropic tool format."""
    anthropic_tools = []
    for tool in tools:
        schema = tool.inputSchema if hasattr(tool, "inputSchema") else {}
        description = getattr(tool, "description", None) or tool.name
        anthropic_tools.append({
            "name": tool.name,
            "description": description,
            "input_schema": schema or {"type": "object", "properties": {}},
        })
    return anthropic_tools


def _call_tool_result_to_text(result: Any) -> str:
    """Extract text from an MCP CallToolResult.

    Preserves empty strings: a tool that returns ``TextContent(text="")``
    is returning an explicit empty result, not "no content".  Only fall
    back to ``str(result)`` (a noisy repr) when there are genuinely no
    text-bearing content blocks at all.
    """
    content = getattr(result, "content", [])
    parts = []
    for c in content:
        text = getattr(c, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts) if parts else str(result)


@dataclass
class _AnthropicHandle:
    """Holds the Anthropic client and conversation state."""
    client: Any
    system_prompt: str
    model: str
    max_tokens: int
    tools: list[dict[str, Any]]
    mcp_server_map: dict[str, Any]  # tool_name -> MCP server handle
    model_settings: dict[str, Any] = field(default_factory=dict)
    stream_thinking: bool = False
    exclude_from_context: bool = False


class AnthropicSDKBackend:
    """Adapter that drives the Anthropic Python SDK."""

    name = "anthropic_sdk"

    def validate(self, spec: AgentSpec) -> None:
        if spec.handoffs or spec.in_handoff_graph:
            raise BackendCapabilityError(
                "anthropic_sdk: agent handoffs are not supported"
            )
        if not spec.model:
            raise BackendBadRequestError("anthropic_sdk: model is required")

    async def build(
        self,
        spec: AgentSpec,
        *,
        run_hooks: Any = None,
        agent_hooks: Any = None,
    ) -> _AnthropicHandle:
        del run_hooks, agent_hooks

        import anthropic

        from ...capi import get_AI_endpoint, get_AI_token, get_provider

        # Resolve token: per-model env var override, then standard token chain.
        # Wrap RuntimeError from get_AI_token (env var not set) so the runner
        # surfaces it as a request error rather than an internal exception.
        token = os.getenv(spec.token_env, "") if spec.token_env else ""
        if not token:
            try:
                token = get_AI_token()
            except RuntimeError as exc:
                raise BackendBadRequestError(
                    f"anthropic_sdk: no API token available ({exc})"
                ) from exc
        if not token:
            raise BackendBadRequestError(
                "anthropic_sdk: no API token available "
                "(checked spec.token_env then standard token chain)"
            )

        endpoint = spec.endpoint or get_AI_endpoint()
        provider = get_provider(endpoint)

        # Providers with bearer_auth=True need Authorization: Bearer instead
        # of the Anthropic SDK's native x-api-key header. Use a placeholder
        # api_key so the SDK doesn't also send the real token via x-api-key.
        # Endpoints not in the provider registry default to native SDK auth.
        headers: dict[str, str] = dict(provider.extra_headers)
        if provider.bearer_auth:
            headers["Authorization"] = f"Bearer {token}"

        client = anthropic.AsyncAnthropic(
            api_key="placeholder" if provider.bearer_auth else token,
            base_url=endpoint,
            default_headers=headers or None,
        )

        # Collect tools from MCP servers and apply blocked_tools filter.
        # We get raw tool lists via list_tools_unfiltered() rather than
        # list_tools(), which would require run_context/agent args to
        # invoke the openai-agents tool_filter -- args we don't have
        # outside the openai-agents run loop.
        #
        # blocked_tools in taskflow YAML are raw (un-namespaced) names,
        # consistent with how openai_agents and copilot_sdk consume them.
        # list_tools_unfiltered() returns namespace-prefixed names (the
        # MCP server wrapper applies the prefix). Match against both
        # forms so blocking works regardless of which name the taskflow
        # author used; key mcp_server_map by the namespaced name because
        # that's what Anthropic will send back in tool_use blocks.
        all_tools: list[dict[str, Any]] = []
        mcp_server_map: dict[str, Any] = {}
        blocked = set(spec.blocked_tools or [])

        def _is_blocked(tool: Any, namespace: str) -> bool:
            name = tool.name
            if name in blocked:
                return True
            return name.startswith(namespace) and name[len(namespace):] in blocked

        for mcp_spec in spec.mcp_servers:
            native_server = mcp_spec.params.get("_native")
            if native_server is None:
                continue
            try:
                mcp_tools = await native_server.list_tools_unfiltered()
                namespace = getattr(native_server, "namespace", "")
                kept = [t for t in mcp_tools if not _is_blocked(t, namespace)]
                for tool in kept:
                    mcp_server_map[tool.name] = native_server
                all_tools.extend(_mcp_tools_to_anthropic(kept))
            except Exception:
                logger.exception("Failed to list tools from MCP server %s", mcp_spec.name)

        # Resolve max_tokens from model_settings or default
        max_tokens = spec.model_settings.get("max_tokens", 16384)
        stream_thinking = spec.model_settings.get("stream_thinking", False)

        return _AnthropicHandle(
            client=client,
            system_prompt=spec.instructions or "",
            model=spec.model,
            max_tokens=max_tokens,
            tools=all_tools,
            mcp_server_map=mcp_server_map,
            model_settings=spec.model_settings,
            stream_thinking=stream_thinking,
            exclude_from_context=spec.exclude_from_context,
        )

    async def run_streamed(
        self,
        agent: Any,
        prompt: str,
        *,
        max_turns: int,
    ) -> AsyncIterator[StreamEvent]:
        handle: _AnthropicHandle = agent
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]

        # Build optional params
        create_kwargs: dict[str, Any] = {}

        # Pass through temperature/top_p if set
        temperature = handle.model_settings.get("temperature")
        if temperature is not None:
            create_kwargs["temperature"] = float(temperature)
        top_p = handle.model_settings.get("top_p")
        if top_p is not None:
            create_kwargs["top_p"] = float(top_p)

        reasoning = handle.model_settings.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if effort:
                if effort not in _VALID_REASONING:
                    raise BackendBadRequestError(
                        f"anthropic_sdk: invalid reasoning effort {effort!r} "
                        f"(expected one of {_VALID_REASONING})"
                    )
                create_kwargs["thinking"] = {"type": "adaptive"}
                create_kwargs["output_config"] = {"effort": effort}

        # Prompt caching: mark the stable prefix (tool definitions + system
        # prompt) with a block-level ``cache_control`` breakpoint. Taskflows
        # reuse the same instructions and tools across many templated prompts,
        # so the first request writes that prefix to cache and every later
        # request reads it back -- a large token-cost reduction on repeated
        # prompts.
        #
        # We use block-level ``cache_control`` (on content blocks) rather than
        # the top-level request param. CAPI's native ``/v1/messages`` surface
        # is a passthrough: its Anthropic model endpoints reject a top-level
        # ``cache_control`` ("Extra inputs are not permitted") while accepting
        # the block-level form on every model -- caching where the upstream
        # supports it and silently ignoring it where it does not. Default on;
        # ``prompt_caching: false`` opts out entirely.
        prompt_caching = handle.model_settings.get("prompt_caching", True)
        system_param: Any = handle.system_prompt
        tools_param: list[dict[str, Any]] = handle.tools
        if prompt_caching:
            ttl = prompt_caching if isinstance(prompt_caching, str) else "5m"
            cache_block: dict[str, Any] = {"type": "ephemeral"}
            if ttl != "5m":
                cache_block["ttl"] = ttl
            if handle.system_prompt:
                system_param = [
                    {
                        "type": "text",
                        "text": handle.system_prompt,
                        "cache_control": cache_block,
                    }
                ]
            if tools_param:
                tools_param = [
                    *tools_param[:-1],
                    {**tools_param[-1], "cache_control": cache_block},
                ]

        import anthropic

        for turn in range(max_turns):
            try:
                async with handle.client.messages.stream(
                    model=handle.model,
                    max_tokens=handle.max_tokens,
                    system=system_param,
                    messages=messages,
                    tools=tools_param or anthropic.NOT_GIVEN,
                    **create_kwargs,
                ) as stream:
                    async for event in stream:
                        if hasattr(event, "type"):
                            if event.type == "content_block_delta":
                                delta = event.delta
                                if hasattr(delta, "text"):
                                    yield TextDelta(text=delta.text)
                                elif hasattr(delta, "thinking") and handle.stream_thinking:
                                    yield TextDelta(text=delta.thinking)

                    response = await stream.get_final_message()

            except anthropic.RateLimitError as exc:
                raise BackendRateLimitError(str(exc)) from exc
            except anthropic.APITimeoutError as exc:
                raise BackendTimeoutError(str(exc)) from exc
            except anthropic.APIStatusError as exc:
                # Map all 4xx (auth, permission, not_found, conflict,
                # unprocessable, bad_request) to BackendBadRequestError so
                # the runner surfaces them as request errors rather than
                # internal exceptions. 5xx and unclassified errors fall
                # through to BackendUnexpectedError.
                status = getattr(exc, "status_code", None)
                if isinstance(status, int) and 400 <= status < 500:
                    raise BackendBadRequestError(str(exc)) from exc
                raise BackendUnexpectedError(str(exc)) from exc
            except anthropic.APIError as exc:
                raise BackendUnexpectedError(str(exc)) from exc

            _emit = getattr(response, "usage", None)
            if _emit is not None:
                yield TokenUsage(
                    model=handle.model,
                    input_tokens=int(getattr(_emit, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(_emit, "output_tokens", 0) or 0),
                    cache_read_tokens=int(getattr(_emit, "cache_read_input_tokens", 0) or 0),
                    cache_write_tokens=int(getattr(_emit, "cache_creation_input_tokens", 0) or 0),
                )

            if response.stop_reason == "end_turn":
                return
            if response.stop_reason != "tool_use":
                return

            # Process tool calls
            tool_use_blocks = [
                b for b in response.content if b.type == "tool_use"
            ]
            if not tool_use_blocks:
                return

            # Add assistant message with all content blocks
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call and collect results
            tool_results: list[dict[str, Any]] = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input

                server = handle.mcp_server_map.get(tool_name)
                if server is None:
                    logger.warning("Tool %s not found in MCP servers", tool_name)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": f"Error: tool '{tool_name}' not found",
                        "is_error": True,
                    })
                    yield ToolEnd(tool_name=tool_name, text=f"Error: tool '{tool_name}' not found")
                    continue

                try:
                    result = await server.call_tool(
                        tool_name,
                        arguments=tool_input if isinstance(tool_input, dict) else {},
                    )
                    result_text = _call_tool_result_to_text(result)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": result_text,
                    })
                    yield ToolEnd(tool_name=tool_name, text=result_text)
                except Exception as exc:
                    logger.exception("Tool call %s failed", tool_name)
                    error_text = f"Error calling {tool_name}: {exc}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": error_text,
                        "is_error": True,
                    })
                    yield ToolEnd(tool_name=tool_name, text=error_text)

            # exclude_from_context: stop after tool results are emitted
            # so they are available to the runner but not fed back into
            # the model context (matches copilot_sdk behavior).
            if handle.exclude_from_context:
                return

            messages.append({"role": "user", "content": tool_results})

        raise BackendMaxTurnsError(f"Exceeded max_turns ({max_turns})")

    async def aclose(self, agent: Any) -> None:
        handle: _AnthropicHandle = agent
        if handle is not None and handle.client is not None:
            await handle.client.close()
