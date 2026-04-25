# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Prompt building logic for Taskflow.

Handles Jinja2 templating and prompt iteration based on MCP tool results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import jinja2

from .available_tools import AvailableTools
from .render_utils import render_model_output
from .template_utils import render_template


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
