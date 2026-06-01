# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Pydantic models for the seclab-taskflow-agent grammar.

These models formally define the YAML grammar for taskflows, personalities,
toolboxes, model configs, and prompts. They provide validation at parse time
while maintaining full backwards compatibility with existing YAML files.
"""

from __future__ import annotations

__all__ = [
    "ApiType",
    "BackendSdk",
    "DOCUMENT_MODELS",
    "ModelConfigDocument",
    "PersonalityDocument",
    "PromptDocument",
    "SUPPORTED_VERSION",
    "ServerParams",
    "TaskDefinition",
    "TaskWrapper",
    "TaskflowDocument",
    "TaskflowHeader",
    "ToolboxDocument",
]

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Valid API type values for model configuration.
ApiType = Literal["chat_completions", "responses"]

# Valid backend names. Must stay in sync with ``sdk._KNOWN``.
BackendSdk = Literal["openai_agents", "copilot_sdk"]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

SUPPORTED_VERSION = "1.0"


class TaskflowHeader(BaseModel):
    """The ``seclab-taskflow-agent`` header block present in every YAML file."""

    model_config = ConfigDict(populate_by_name=True)

    version: str
    filetype: str

    @field_validator("version", mode="before")
    @classmethod
    def _normalise_version(cls, v: Any) -> str:
        """Accept int/float/str versions and normalise to ``"1.0"`` format."""
        if isinstance(v, int):
            return f"{v}.0"
        if isinstance(v, float):
            return str(v)
        return str(v)

    @field_validator("version", mode="after")
    @classmethod
    def _validate_version(cls, v: str) -> str:
        if v != SUPPORTED_VERSION:
            msg = f"Unsupported version: {v}. Only version {SUPPORTED_VERSION} is supported."
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# Task definition (a single step inside a taskflow)
# ---------------------------------------------------------------------------

class TaskDefinition(BaseModel):
    """A single task within a taskflow.

    This captures every field the engine currently recognises in a task block.
    Extra fields are allowed for forward-compatibility.
    """

    model_config = ConfigDict(extra="allow")

    name: str = ""
    description: str = ""
    agents: list[str] = Field(default_factory=list)
    user_prompt: str = ""
    run: str = ""
    model: str = ""
    model_settings: dict[str, Any] = Field(default_factory=dict)
    must_complete: bool = False
    headless: bool = False
    repeat_prompt: bool = False
    exclude_from_context: bool = False
    blocked_tools: list[str] = Field(default_factory=list)
    toolboxes: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = 0  # 0 means use the runner default
    uses: str = ""

    # async settings (``async`` is a reserved word, aliased)
    async_task: bool = Field(default=False, alias="async")
    async_limit: int = 5

    @model_validator(mode="after")
    def _run_xor_prompt(self) -> TaskDefinition:
        if self.run and self.user_prompt:
            raise ValueError("shell task ('run') and prompt task ('user_prompt') are mutually exclusive")
        return self


class TaskWrapper(BaseModel):
    """Wraps the ``- task:`` YAML list entry."""

    task: TaskDefinition


# ---------------------------------------------------------------------------
# Top-level document types
# ---------------------------------------------------------------------------

class TaskflowDocument(BaseModel):
    """A complete taskflow YAML document.

    Example::

        seclab-taskflow-agent:
          version: "1.0"
          filetype: taskflow
        globals:
          fruit: bananas
        model_config_ref: examples.model_configs.model_config
        taskflow:
          - task:
              ...
    """

    model_config = ConfigDict(extra="allow")

    header: TaskflowHeader = Field(alias="seclab-taskflow-agent")
    globals: dict[str, Any] = Field(default_factory=dict)
    # ``model_config`` clashes with Pydantic's own ConfigDict, so we use an alias
    model_config_ref: str = Field(default="", alias="model_config")
    taskflow: list[TaskWrapper] = Field(default_factory=list)

    @field_validator("taskflow", mode="before")
    @classmethod
    def _coerce_taskflow_list(cls, v: Any) -> list[Any]:
        if v is None:
            return []
        return v


class PersonalityDocument(BaseModel):
    """A personality YAML document."""

    model_config = ConfigDict(extra="allow")

    header: TaskflowHeader = Field(alias="seclab-taskflow-agent")
    personality: str = ""
    task: str = ""
    toolboxes: list[str] = Field(default_factory=list)


class ServerParams(BaseModel):
    """MCP server connection parameters inside a toolbox."""

    model_config = ConfigDict(extra="allow")

    kind: str
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    optional_headers: dict[str, str] | None = None
    timeout: float | None = None
    reconnecting: bool = False


class ToolboxDocument(BaseModel):
    """A toolbox YAML document defining an MCP server configuration."""

    model_config = ConfigDict(extra="allow")

    header: TaskflowHeader = Field(alias="seclab-taskflow-agent")
    server_params: ServerParams
    server_prompt: str = ""
    confirm: list[str] = Field(default_factory=list)
    client_session_timeout: float = 0


class ModelConfigDocument(BaseModel):
    """A model_config YAML document mapping logical model names to provider IDs.

    The ``api_type`` field controls which OpenAI API is used for all models
    in this config: ``"chat_completions"`` (default) or ``"responses"``.

    The ``backend`` field selects which SDK adapter drives the agent loop.
    When unset the runner falls back to ``SECLAB_TASKFLOW_BACKEND`` or an
    endpoint-based auto-default. Unknown values are rejected at parse time.
    """

    model_config = ConfigDict(extra="allow")

    header: TaskflowHeader = Field(alias="seclab-taskflow-agent")
    api_type: ApiType = "chat_completions"
    backend: BackendSdk | None = None
    models: dict[str, str] = Field(default_factory=dict)
    model_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PromptDocument(BaseModel):
    """A reusable prompt YAML document."""

    model_config = ConfigDict(extra="allow")

    header: TaskflowHeader = Field(alias="seclab-taskflow-agent")
    prompt: str = ""


# ---------------------------------------------------------------------------
# Mapping from filetype string → Pydantic model
# ---------------------------------------------------------------------------

DOCUMENT_MODELS: dict[str, type[BaseModel]] = {
    "taskflow": TaskflowDocument,
    "personality": PersonalityDocument,
    "toolbox": ToolboxDocument,
    "model_config": ModelConfigDocument,
    "prompt": PromptDocument,
}
