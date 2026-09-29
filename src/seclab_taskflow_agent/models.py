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
    "CompletionPolicy",
    "DOCUMENT_MODELS",
    "ModelConfigDocument",
    "ModelEntry",
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
ApiType = Literal["chat_completions", "responses", "messages"]

# Valid backend names. Must stay in sync with ``sdk._KNOWN``.
BackendSdk = Literal["openai_agents", "copilot_sdk", "anthropic_sdk"]

# Completion policy for a task's fan-out: whether every branch (each prompt x
# model cell) must succeed for the task to be considered complete, or any
# single branch succeeding is enough.
CompletionPolicy = Literal["all", "any"]

# Which produced value a task captures as its named ``outputs.<id>``:
# ``tool_result`` (default) captures the task's final tool result; ``response``
# captures the agent's final response text (the prose after its last tool
# call), which is what model-comparison / evaluation flows want.
CaptureSource = Literal["tool_result", "response"]


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

class ModelEntry(BaseModel):
    """A single model entry for multi-model task execution.

    Accepts either a bare logical model name (coerced to ``model=<name>``)
    or a mapping with ``model`` and optional ``model_settings``. Logical
    names are resolved through the taskflow's ``model_config`` exactly like
    the singular ``model:`` field, so per-entry ``api_type``/``endpoint``/
    ``token``/``backend`` overrides work the same way.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str = ""
    model_settings: dict[str, Any] = Field(default_factory=dict)


class TaskDefinition(BaseModel):
    """A single task within a taskflow.

    This captures every field the engine currently recognises in a task block.
    Extra fields are allowed for forward-compatibility.
    """

    model_config = ConfigDict(extra="allow")

    name: str = ""
    description: str = ""
    id: str = ""
    # GitHub-Actions-style conditional: when set, the task runs only if this
    # Jinja expression evaluates truthy against the template context
    # (globals / inputs / outputs). ``if`` is a Python keyword, so it is
    # aliased. Empty means the task always runs.
    if_: str = Field(default="", alias="if")
    agents: list[str] = Field(default_factory=list)
    user_prompt: str = ""
    run: str = ""
    model: str = ""
    model_settings: dict[str, Any] = Field(default_factory=dict)
    # Typed named outputs: declare a JSON Schema for this task's produced
    # value; when set, the captured output is validated and exposed to
    # later tasks as ``outputs.<id>``.
    outputs: dict[str, Any] = Field(default_factory=dict)
    # Which produced value feeds ``outputs.<id>``: the task's final tool result
    # (default) or the agent's final response text (``response``). Response
    # capture is what side-by-side model comparison / evaluation flows want,
    # since the value to compare is the model's prose answer, not a tool result.
    capture: CaptureSource = "tool_result"
    # Explicit iterable selector for repeat_prompt: a Jinja expression
    # evaluated against the template context (e.g. ``outputs.list_fns.items``).
    over: str = ""
    # Multi-model fan-out: run this task against each listed model in
    # parallel with per-model output streams. Mutually exclusive with the
    # singular ``model`` field. Empty means single-model (see ``model``).
    models: list[ModelEntry] = Field(default_factory=list)
    # Completion policy for the task's fan-out (all prompt x model branches):
    # "all" (default) requires every branch to succeed; "any" needs one.
    completion: CompletionPolicy = "all"
    # Upper bound on concurrent branch runs for a multi-model task
    # (0 = run all models at once). Does not affect single-model tasks.
    model_concurrency: int = 0
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

    @field_validator("models", mode="before")
    @classmethod
    def _coerce_models(cls, v: Any) -> list[Any]:
        """Coerce list items into ``ModelEntry`` maps.

        Accepts a list whose entries are either bare model-name strings
        (``[gpt_default, claude_native]``) or ``{model, model_settings}``
        maps. Strings become ``{"model": <name>}``; maps and ``ModelEntry``
        instances pass through for ``ModelEntry`` to validate.
        """
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError(
                "'models' must be a list of model names or {model, model_settings} maps"
            )
        out: list[Any] = []
        for item in v:
            if isinstance(item, str):
                out.append({"model": item})
            elif isinstance(item, (dict, ModelEntry)):
                out.append(item)
            else:
                raise ValueError(f"invalid 'models' entry: {item!r}")
        return out

    @model_validator(mode="after")
    def _run_xor_prompt(self) -> TaskDefinition:
        if self.run and self.user_prompt:
            raise ValueError("shell task ('run') and prompt task ('user_prompt') are mutually exclusive")
        if self.capture == "response" and self.run:
            raise ValueError(
                "capture: response captures an agent's final response text and does not "
                "apply to a shell task ('run'); use the default capture: tool_result"
            )
        return self

    @model_validator(mode="after")
    def _validate_models(self) -> TaskDefinition:
        if self.models and self.model:
            raise ValueError(
                "'model' and 'models' are mutually exclusive; use 'model' for a single "
                "model or 'models' for multi-model fan-out"
            )
        if self.model_concurrency < 0:
            raise ValueError("'model_concurrency' must be >= 0")
        return self

    @model_validator(mode="after")
    def _validate_outputs(self) -> TaskDefinition:
        # Fail fast on a malformed outputs schema at load time, before any
        # model calls are made.
        if self.outputs:
            from .output_schema import OutputSchemaError, validate_output_schema

            try:
                validate_output_schema(self.outputs)
            except OutputSchemaError as exc:
                # OutputSchemaError already carries a fully-formed, descriptive
                # message; surface it directly rather than re-prefixing (which
                # would duplicate its "invalid 'outputs' schema:" prefix).
                raise ValueError(str(exc)) from exc
        if self.over and not self.repeat_prompt:
            raise ValueError("'over' only applies to repeat_prompt tasks")
        return self

    def effective_model_entries(self) -> list[ModelEntry]:
        """Return the normalised list of models this task runs against.

        When ``models`` is set it is returned as-is. Otherwise the singular
        ``model``/``model_settings`` pair is wrapped in a one-element list so
        callers can treat single-model and multi-model tasks uniformly.
        """
        if self.models:
            return list(self.models)
        return [ModelEntry(model=self.model, model_settings=dict(self.model_settings))]


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
