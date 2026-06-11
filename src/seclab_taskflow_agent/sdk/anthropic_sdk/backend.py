# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Anthropic SDK backend adapter.

Drives the Anthropic Messages API (``/v1/messages``) via the official
``anthropic`` Python SDK. Supports streaming, tool calling via MCP
servers, and extended thinking.

Auth note: The Anthropic SDK sends ``x-api-key`` by default, but CAPI
expects ``Authorization: Bearer``. We pass the bearer header via
``default_headers`` and set ``api_key`` to a placeholder so the SDK
doesn't complain about a missing key.
"""

from __future__ import annotations

__all__ = ["AnthropicSDKBackend"]

import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..base import AgentSpec, StreamEvent, TextDelta, ToolEnd
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


def _resolve_token(token_env: str | None) -> str:
    """Resolve the API token from env var name or default AI_API_TOKEN."""
    if token_env:
        val = os.getenv(token_env)
        if val:
            return val
    val = os.getenv("AI_API_TOKEN")
    if val:
        return val
    raise BackendBadRequestError(
        "anthropic_sdk: no API token found (set AI_API_TOKEN or per-model token env)"
    )


def _resolve_endpoint() -> str:
    """Resolve the API base URL."""
    return os.getenv("AI_API_ENDPOINT", "https://api.githubcopilot.com")


def _mcp_tools_to_anthropic(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to Anthropic tool format."""
    anthropic_tools = []
    for tool in tools:
        schema = tool.inputSchema if hasattr(tool, "inputSchema") else {}
        anthropic_tools.append({
            "name": tool.name,
            "description": getattr(tool, "description", tool.name),
            "input_schema": schema or {"type": "object", "properties": {}},
        })
    return anthropic_tools


def _call_tool_result_to_text(result: Any) -> str:
    """Extract text from an MCP CallToolResult."""
    content = getattr(result, "content", [])
    parts = []
    for c in content:
        text = getattr(c, "text", None)
        if text:
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

        token = _resolve_token(spec.token_env)
        endpoint = spec.endpoint or _resolve_endpoint()

        client = anthropic.AsyncAnthropic(
            api_key="placeholder",
            base_url=endpoint,
            default_headers={
                "Authorization": f"Bearer {token}",
                "Copilot-Integration-Id": os.getenv(
                    "COPILOT_INTEGRATION_ID", "vscode-chat"
                ),
            },
        )

        # Collect tools from MCP servers
        all_tools: list[dict[str, Any]] = []
        mcp_server_map: dict[str, Any] = {}

        for mcp_spec in spec.mcp_servers:
            native_server = mcp_spec.params.get("_native")
            if native_server is None:
                continue
            try:
                mcp_tools = await native_server.list_tools()
                anthropic_tools = _mcp_tools_to_anthropic(mcp_tools)
                all_tools.extend(anthropic_tools)
                for tool in mcp_tools:
                    mcp_server_map[tool.name] = native_server
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

        import anthropic

        for turn in range(max_turns):
            try:
                async with handle.client.messages.stream(
                    model=handle.model,
                    max_tokens=handle.max_tokens,
                    system=handle.system_prompt,
                    messages=messages,
                    tools=handle.tools or anthropic.NOT_GIVEN,
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
            except anthropic.BadRequestError as exc:
                raise BackendBadRequestError(str(exc)) from exc
            except anthropic.APIError as exc:
                raise BackendUnexpectedError(str(exc)) from exc

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
