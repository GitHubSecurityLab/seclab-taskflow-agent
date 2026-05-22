# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""SecLab Taskflow Agent — secure and automated workflow execution.

This package provides the engine for running declarative YAML-based taskflows
that orchestrate AI agents with MCP (Model Context Protocol) tool servers
for security analysis, code auditing, and vulnerability triage.

Architecture
~~~~~~~~~~~~
- :mod:`~seclab_taskflow_agent.models` — Pydantic v2 grammar models
- :mod:`~seclab_taskflow_agent.available_tools` — YAML resource loader & cache
- :mod:`~seclab_taskflow_agent.cli` — CLI entry point (Typer)
- :mod:`~seclab_taskflow_agent.runner` — Taskflow execution engine
- :mod:`~seclab_taskflow_agent.agent` — Agent / hooks wrapper classes
- :mod:`~seclab_taskflow_agent.mcp_lifecycle` — MCP server connect / cleanup
- :mod:`~seclab_taskflow_agent.mcp_utils` — MCP client parameter resolution
- :mod:`~seclab_taskflow_agent.mcp_transport` — MCP transport implementations
- :mod:`~seclab_taskflow_agent.mcp_prompt` — System prompt construction
- :mod:`~seclab_taskflow_agent.session` — Taskflow checkpoint / resume
- :mod:`~seclab_taskflow_agent.template_utils` — Jinja2 template rendering
- :mod:`~seclab_taskflow_agent.prompt_parser` — Legacy prompt argument parser
"""

__all__ = [
    "ApiType",
    "AvailableTools",
    "BackendSdk",
    "TaskAgent",
    "TaskRunHooks",
    "TaskAgentHooks",
    "PersonalityDocument",
    "TaskflowDocument",
    "ToolboxDocument",
    "ModelConfigDocument",
    "PromptDocument",
    "TaskDefinition",
]

from .agent import TaskAgent, TaskAgentHooks, TaskRunHooks
from .available_tools import AvailableTools
from .models import (
    ApiType,
    BackendSdk,
    ModelConfigDocument,
    PersonalityDocument,
    PromptDocument,
    TaskDefinition,
    TaskflowDocument,
    ToolboxDocument,
)
