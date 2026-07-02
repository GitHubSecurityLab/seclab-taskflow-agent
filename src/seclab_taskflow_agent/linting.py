# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Offline taskflow linter.

Validates a taskflow and every document it references without making any
model calls, turning what would otherwise be an expensive runtime failure
into an instant check. Surfaces:

* grammar/parse errors (via the loader),
* unknown fields (typos) - reported as warnings, or errors in ``strict`` mode,
* missing referenced personalities / toolboxes / reusable taskflows,
* logical model names that are not defined in the resolved ``model_config``,
* malformed Jinja in prompts / ``over`` expressions.

The core :func:`lint_taskflow` is a pure function returning a list of
:class:`LintIssue`, so it is fully unit-testable offline.
"""

from __future__ import annotations

__all__ = ["LintIssue", "format_issues", "lint_taskflow"]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jinja2

from .models import ModelEntry, TaskDefinition
from .template_utils import create_jinja_environment

if TYPE_CHECKING:
    from .available_tools import AvailableTools


@dataclass
class LintIssue:
    """A single problem found while linting a taskflow."""

    severity: str  # "error" | "warning"
    code: str
    message: str
    location: str = ""


@dataclass
class _Ctx:
    """Shared lint state."""

    available_tools: AvailableTools
    strict: bool
    model_dict: dict[str, str]
    has_model_config: bool
    issues: list[LintIssue] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, location: str = "") -> None:
        self.issues.append(LintIssue(severity, code, message, location))


def _check_unknown_fields(model: object, location: str, ctx: _Ctx) -> None:
    """Report ``extra="allow"`` overflow fields (likely typos)."""
    extra = getattr(model, "model_extra", None) or {}
    for key in extra:
        severity = "error" if ctx.strict else "warning"
        ctx.add(severity, "unknown-field", f"unknown field {key!r}", location)


def _check_reference(loader, name: str, kind: str, location: str, ctx: _Ctx) -> bool:
    """Verify a referenced document loads; report an error if not."""
    try:
        loader(name)
    except Exception as exc:  # noqa: BLE001 - loader raises several types
        ctx.add("error", f"missing-{kind}", f"cannot load {kind} {name!r}: {exc}", location)
        return False
    return True


def _check_template(template_str: str, what: str, location: str, ctx: _Ctx) -> None:
    """Report malformed Jinja syntax in a template (does not render)."""
    if not template_str:
        return
    env = create_jinja_environment(ctx.available_tools)
    try:
        env.from_string(template_str)
    except jinja2.TemplateSyntaxError as exc:
        ctx.add("error", "template-syntax", f"invalid Jinja in {what}: {exc}", location)


def _check_expression(expression: str, what: str, location: str, ctx: _Ctx) -> None:
    """Report malformed Jinja in an *expression* (e.g. ``over``).

    ``over`` is evaluated at runtime with ``compile_expression`` (matching
    ``template_utils.evaluate_expression``), so it must be checked as an
    expression, not a template - otherwise a bare ``globals.items`` looks like
    literal text and no syntax error is ever reported.
    """
    if not expression:
        return
    expr = expression.strip()
    if expr.startswith("{{") and expr.endswith("}}"):
        expr = expr[2:-2].strip()
    env = create_jinja_environment(ctx.available_tools)
    try:
        env.compile_expression(expr)
    except jinja2.TemplateSyntaxError as exc:
        ctx.add("error", "template-syntax", f"invalid Jinja in {what}: {exc}", location)


def _check_model_name(name: str, location: str, ctx: _Ctx) -> None:
    """Warn when a logical model name is not defined in the model_config."""
    if not name:
        return
    if ctx.has_model_config and name not in ctx.model_dict:
        ctx.add(
            "warning",
            "unknown-model",
            f"model {name!r} is not defined in the model_config; it will be used "
            f"as a literal provider id (typo?)",
            location,
        )


def _lint_task(task: TaskDefinition, location: str, ctx: _Ctx) -> None:
    at = ctx.available_tools
    _check_unknown_fields(task, location, ctx)

    # Reusable taskflow reference.
    if task.uses:
        try:
            reusable = at.get_taskflow(task.uses)
            if len(reusable.taskflow) != 1:
                ctx.add(
                    "error",
                    "reusable-shape",
                    f"reusable taskflow {task.uses!r} must contain exactly 1 task",
                    location,
                )
        except Exception as exc:  # noqa: BLE001
            ctx.add("error", "missing-uses", f"cannot load reusable taskflow {task.uses!r}: {exc}", location)

    # Agents (personalities) and their toolboxes.
    for agent_name in task.agents or []:
        if _check_reference(at.get_personality, agent_name, "personality", location, ctx):
            personality = at.get_personality(agent_name)
            if not (task.toolboxes or []):
                for tb in personality.toolboxes or []:
                    _check_reference(at.get_toolbox, tb, "toolbox", location, ctx)

    # Task-level toolbox overrides.
    for tb in task.toolboxes or []:
        _check_reference(at.get_toolbox, tb, "toolbox", location, ctx)

    # Model name resolution.
    entries: list[ModelEntry] = task.effective_model_entries()
    for entry in entries:
        _check_model_name(entry.model, location, ctx)

    # Prompt / over template syntax.
    _check_template(task.user_prompt, "user_prompt", location, ctx)
    _check_expression(task.over, "over", location, ctx)


def lint_taskflow(
    available_tools: AvailableTools,
    taskflow_path: str,
    *,
    strict: bool = False,
    cli_model_config: str | None = None,
) -> list[LintIssue]:
    """Lint a taskflow and every document it references, offline.

    Args:
        available_tools: Tool registry / loader.
        taskflow_path: Dotted path of the taskflow to lint.
        strict: Treat unknown fields as errors instead of warnings.
        cli_model_config: Overrides the taskflow's own ``model_config``.

    Returns:
        A list of :class:`LintIssue`. An empty list means no problems.
    """
    ctx = _Ctx(available_tools=available_tools, strict=strict, model_dict={}, has_model_config=False)

    # Load the taskflow itself (grammar/parse errors surface here).
    try:
        doc = available_tools.get_taskflow(taskflow_path)
    except Exception as exc:  # noqa: BLE001 - loader raises several types
        ctx.add("error", "load", f"cannot load taskflow: {exc}", taskflow_path)
        return ctx.issues

    _check_unknown_fields(doc, taskflow_path, ctx)

    # Resolve the model_config (CLI override wins), if any.
    model_config_ref = cli_model_config or doc.model_config_ref
    if model_config_ref:
        try:
            mc = available_tools.get_model_config(model_config_ref)
            ctx.model_dict = mc.models or {}
            ctx.has_model_config = True
        except Exception as exc:  # noqa: BLE001
            ctx.add("error", "model_config", f"cannot load model_config {model_config_ref!r}: {exc}", taskflow_path)

    for i, wrapper in enumerate(doc.taskflow):
        task = wrapper.task
        location = f"task[{i}]" + (f" ({task.name})" if task.name else "")
        _lint_task(task, location, ctx)

    return ctx.issues


def format_issues(issues: list[LintIssue]) -> str:
    """Render lint issues as human-readable lines."""
    if not issues:
        return "No issues found."
    lines: list[str] = []
    for issue in issues:
        loc = f" [{issue.location}]" if issue.location else ""
        lines.append(f"{issue.severity.upper()}: {issue.code}:{loc} {issue.message}")
    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    lines.append(f"({errors} error(s), {warnings} warning(s))")
    return "\n".join(lines)
