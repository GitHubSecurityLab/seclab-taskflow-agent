# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Neutral cross-task result representation and per-run result store.

Historically the runner passed data between tasks through a single shared
``list[str]`` of backend-specific tool-result JSON envelopes, decoded with a
double ``json.loads`` that only reliably handled the openai-agents
single-text-content shape. The copilot and anthropic adapters had to
*reconstruct* that exact envelope (``json.dumps({"text": ...})``) so the
runner's decode kept working.

This module replaces that with:

* :class:`ToolResult` - a neutral, backend-agnostic record of a single tool
  call result (``text`` and/or ``structured`` content).
* :func:`normalize_openai_tool_output` - turns the openai-agents MCP
  serialization (any of its three shapes) into a :class:`ToolResult`.
* :func:`decode_tool_result` - the single place that turns a
  :class:`ToolResult` into the Python value a task consumes.
* :class:`ResultStore` - a per-run store holding the ordered tool results
  (for legacy ``repeat_prompt`` carry-over) and named task outputs (for typed
  ``outputs.<id>`` passing), with snapshot/restore for session resume.
"""

from __future__ import annotations

__all__ = [
    "ResultStore",
    "ToolResult",
    "decode_tool_result",
    "normalize_openai_tool_output",
]

import json
from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolResult(BaseModel):
    """A neutral record of one tool call result.

    ``text`` carries the primary text payload (when the tool returned text
    content); ``structured`` carries a decoded object when the tool returned
    structured / multi-part content. Exactly which is populated depends on the
    tool and backend; :func:`decode_tool_result` prefers ``structured`` and
    falls back to parsing ``text``.
    """

    model_config = ConfigDict(extra="allow")

    tool_name: str = ""
    text: str | None = None
    structured: Any = None


def normalize_openai_tool_output(raw: Any, tool_name: str = "") -> ToolResult:
    """Normalise an openai-agents MCP tool output into a :class:`ToolResult`.

    The openai-agents MCP shim serialises a tool result as one of:

    * a single text content part: ``{"type": "text", "text": "..."}``
    * multiple content parts: ``[{...}, {...}]``
    * structured content: an arbitrary JSON object
    * empty content: ``"[]"``

    plus the historical bridge envelope ``{"text": "..."}``. This function
    accepts all of them (and already-normalised inputs) and never raises.
    """
    if isinstance(raw, ToolResult):
        return raw
    if not isinstance(raw, str):
        # Already a structured Python object (e.g. a dict/list).
        return _from_parsed(raw, tool_name)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Not JSON at all - keep it as plain text.
        return ToolResult(tool_name=tool_name, text=raw)
    return _from_parsed(parsed, tool_name)


def _from_parsed(parsed: Any, tool_name: str) -> ToolResult:
    """Build a :class:`ToolResult` from an already-parsed value."""
    if isinstance(parsed, dict):
        # MCP single text content part.
        if parsed.get("type") == "text" and isinstance(parsed.get("text"), str):
            return ToolResult(tool_name=tool_name, text=parsed["text"])
        # Historical bridge envelope: a lone ``text`` key.
        if list(parsed.keys()) == ["text"] and isinstance(parsed.get("text"), str):
            return ToolResult(tool_name=tool_name, text=parsed["text"])
        # Anything else is structured content.
        return ToolResult(tool_name=tool_name, structured=parsed)
    if isinstance(parsed, list):
        texts = [
            item.get("text")
            for item in parsed
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        # Pure list of text parts -> join as text; otherwise keep structured.
        if texts and len(texts) == len(parsed):
            return ToolResult(tool_name=tool_name, text="\n".join(texts))
        return ToolResult(tool_name=tool_name, structured=parsed)
    if isinstance(parsed, str):
        return ToolResult(tool_name=tool_name, text=parsed)
    # Scalars (int/float/bool) are treated as structured values.
    return ToolResult(tool_name=tool_name, structured=parsed)


def decode_tool_result(tr: ToolResult) -> Any:
    """Return the Python value a task consumes from a :class:`ToolResult`.

    Prefers ``structured`` content; otherwise parses ``text`` as JSON. Raises
    ``ValueError`` when there is no usable content or the text is not valid
    JSON, matching the strictness the legacy ``repeat_prompt`` path required.
    """
    if tr.structured is not None:
        return tr.structured
    if tr.text is None:
        raise ValueError("tool result has neither text nor structured content")
    try:
        return json.loads(tr.text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("tool result text is not valid JSON") from exc


class ResultStore:
    """Per-run store for cross-task data flow.

    Holds the ordered list of tool results (so legacy ``repeat_prompt`` can
    consume the previous task's last result) and a map of named task outputs
    (so typed ``outputs.<id>`` passing can address a task's result by name).
    Not a module global: one instance per run, threaded through the runner,
    which keeps concurrent (multi-model / async) branches from corrupting a
    shared list.
    """

    def __init__(self) -> None:
        self._results: list[ToolResult] = []
        self._outputs: dict[str, Any] = {}

    def record(self, result: ToolResult) -> None:
        """Append a tool result to the ordered carry-over list."""
        self._results.append(result)

    def last(self) -> ToolResult | None:
        """Return the most recently recorded tool result, or ``None``."""
        return self._results[-1] if self._results else None

    def pop_last(self) -> None:
        """Consume the most recent tool result (legacy repeat_prompt)."""
        if self._results:
            self._results.pop()

    def set_output(self, name: str, value: Any) -> None:
        """Store a named task output for ``outputs.<name>`` addressing."""
        self._outputs[name] = value

    @property
    def outputs(self) -> dict[str, Any]:
        """The named task outputs mapping."""
        return self._outputs

    @property
    def results(self) -> list[ToolResult]:
        """The ordered tool-result carry-over list."""
        return self._results

    def snapshot(self) -> dict[str, Any]:
        """Serialise the store for session persistence."""
        return {
            "results": [r.model_dump() for r in self._results],
            "outputs": self._outputs,
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any] | None) -> ResultStore:
        """Reconstruct a store from :meth:`snapshot` output."""
        store = cls()
        if data:
            store._results = [ToolResult.model_validate(r) for r in data.get("results", [])]
            store._outputs = dict(data.get("outputs", {}))
        return store
