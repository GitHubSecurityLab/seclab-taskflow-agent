# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the offline taskflow linter."""

from unittest.mock import MagicMock

from seclab_taskflow_agent.linting import LintIssue, format_issues, lint_taskflow
from seclab_taskflow_agent.models import (
    ModelConfigDocument,
    PersonalityDocument,
    TaskDefinition,
    TaskflowDocument,
    TaskflowHeader,
    ToolboxDocument,
    TaskWrapper,
)


def _taskflow(tasks, model_config_ref="") -> TaskflowDocument:
    return TaskflowDocument(
        **{
            "seclab-taskflow-agent": TaskflowHeader(version="1.0", filetype="taskflow"),
            "model_config": model_config_ref,
            "taskflow": [TaskWrapper(task=t) for t in tasks],
        }
    )


def _personality(toolboxes=None) -> PersonalityDocument:
    return PersonalityDocument(
        **{
            "seclab-taskflow-agent": TaskflowHeader(version="1.0", filetype="personality"),
            "personality": "p",
            "task": "t",
            "toolboxes": toolboxes or [],
        }
    )


def _model_config(models) -> ModelConfigDocument:
    return ModelConfigDocument(
        **{
            "seclab-taskflow-agent": TaskflowHeader(version="1.0", filetype="model_config"),
            "models": models,
        }
    )


def _toolbox() -> ToolboxDocument:
    return ToolboxDocument(
        **{
            "seclab-taskflow-agent": TaskflowHeader(version="1.0", filetype="toolbox"),
            "server_params": {"kind": "stdio", "command": "echo"},
        }
    )


def _at(doc, *, personality=None, model_config=None, toolbox=None):
    at = MagicMock()
    at.get_taskflow.return_value = doc
    at.get_personality.return_value = personality if personality is not None else _personality()
    if model_config is not None:
        at.get_model_config.return_value = model_config
    if toolbox is not None:
        at.get_toolbox.return_value = toolbox
    else:
        at.get_toolbox.return_value = _toolbox()
    return at


def _codes(issues):
    return {i.code for i in issues}


class TestLintTaskflow:
    def test_clean_taskflow_no_issues(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="hi")])
        at = _at(doc)
        assert lint_taskflow(at, "pkg.flow") == []

    def test_unknown_field_is_warning(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="hi", must_complte=True)])
        at = _at(doc)
        issues = lint_taskflow(at, "pkg.flow")
        assert any(i.code == "unknown-field" and i.severity == "warning" for i in issues)

    def test_unknown_field_is_error_in_strict(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="hi", typo_field=1)])
        at = _at(doc)
        issues = lint_taskflow(at, "pkg.flow", strict=True)
        assert any(i.code == "unknown-field" and i.severity == "error" for i in issues)

    def test_missing_personality_is_error(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.missing"], user_prompt="hi")])
        at = _at(doc)
        at.get_personality.side_effect = Exception("nope")
        issues = lint_taskflow(at, "pkg.flow")
        assert any(i.code == "missing-personality" and i.severity == "error" for i in issues)

    def test_missing_toolbox_override_is_error(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="hi", toolboxes=["pkg.badtb"])])
        at = _at(doc)
        at.get_toolbox.side_effect = Exception("no toolbox")
        issues = lint_taskflow(at, "pkg.flow")
        assert any(i.code == "missing-toolbox" for i in issues)

    def test_personality_toolbox_checked_when_no_override(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="hi")])
        at = _at(doc, personality=_personality(toolboxes=["pkg.ptb"]))
        at.get_toolbox.side_effect = Exception("missing")
        issues = lint_taskflow(at, "pkg.flow")
        assert any(i.code == "missing-toolbox" for i in issues)

    def test_unknown_model_is_warning(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="hi", model="typo")], model_config_ref="pkg.mc")
        at = _at(doc, model_config=_model_config({"fast": "gpt-4o"}))
        issues = lint_taskflow(at, "pkg.flow")
        assert any(i.code == "unknown-model" for i in issues)

    def test_known_model_no_warning(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="hi", model="fast")], model_config_ref="pkg.mc")
        at = _at(doc, model_config=_model_config({"fast": "gpt-4o"}))
        issues = lint_taskflow(at, "pkg.flow")
        assert not any(i.code == "unknown-model" for i in issues)

    def test_no_model_config_no_model_warning(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="hi", model="gpt-4o")])
        at = _at(doc)
        issues = lint_taskflow(at, "pkg.flow")
        assert not any(i.code == "unknown-model" for i in issues)

    def test_bad_model_config_is_error(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="hi")], model_config_ref="pkg.badmc")
        at = _at(doc)
        at.get_model_config.side_effect = Exception("bad mc")
        issues = lint_taskflow(at, "pkg.flow")
        assert any(i.code == "model_config" and i.severity == "error" for i in issues)

    def test_template_syntax_error_is_error(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="{{ unclosed")])
        at = _at(doc)
        issues = lint_taskflow(at, "pkg.flow")
        assert any(i.code == "template-syntax" for i in issues)

    def test_reusable_wrong_shape_is_error(self):
        # a task that uses a reusable taskflow with 2 tasks
        reusable = _taskflow([TaskDefinition(name="a"), TaskDefinition(name="b")])
        doc = _taskflow([TaskDefinition(uses="pkg.reusable", user_prompt="hi")])
        at = MagicMock()
        at.get_taskflow.side_effect = lambda name: doc if name == "pkg.flow" else reusable
        at.get_personality.return_value = _personality()
        at.get_toolbox.return_value = _toolbox()
        issues = lint_taskflow(at, "pkg.flow")
        assert any(i.code == "reusable-shape" for i in issues)

    def test_load_failure_short_circuits(self):
        at = MagicMock()
        at.get_taskflow.side_effect = Exception("cannot parse")
        issues = lint_taskflow(at, "pkg.flow")
        assert len(issues) == 1
        assert issues[0].code == "load"

    def test_cli_model_config_override_used(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="hi", model="fast")])
        at = _at(doc, model_config=_model_config({"other": "gpt-4o"}))
        issues = lint_taskflow(at, "pkg.flow", cli_model_config="pkg.override")
        at.get_model_config.assert_called_once_with("pkg.override")
        # "fast" is not in the override config -> warning
        assert any(i.code == "unknown-model" for i in issues)


class TestFormatIssues:
    def test_empty(self):
        assert format_issues([]) == "No issues found."

    def test_counts(self):
        issues = [
            LintIssue("error", "x", "boom", "task[0]"),
            LintIssue("warning", "y", "meh"),
        ]
        out = format_issues(issues)
        assert "ERROR" in out
        assert "WARNING" in out
        assert "1 error(s), 1 warning(s)" in out


class TestLintOverExpression:
    """The linter must check `over` as an expression, not a template."""

    def test_bad_over_expression_is_error(self):
        doc = _taskflow(
            [TaskDefinition(agents=["pkg.p"], user_prompt="{{ result }}", repeat_prompt=True,
                            over="globals.items | badfilter(")]
        )
        at = _at(doc)
        issues = lint_taskflow(at, "pkg.flow")
        assert any(i.code == "template-syntax" for i in issues)

    def test_bare_over_expression_ok(self):
        doc = _taskflow(
            [TaskDefinition(agents=["pkg.p"], user_prompt="{{ result }}", repeat_prompt=True,
                            over="globals.items")]
        )
        at = _at(doc)
        issues = lint_taskflow(at, "pkg.flow")
        assert not any(i.code == "template-syntax" for i in issues)


class TestLintIfExpression:
    """The linter checks `if` as an expression."""

    def test_bad_if_expression_is_error(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="hi", **{"if": "globals.x =="})])
        at = _at(doc)
        issues = lint_taskflow(at, "pkg.flow")
        assert any(i.code == "template-syntax" for i in issues)

    def test_valid_if_expression_ok(self):
        doc = _taskflow([TaskDefinition(agents=["pkg.p"], user_prompt="hi", **{"if": "globals.x > 3"})])
        at = _at(doc)
        issues = lint_taskflow(at, "pkg.flow")
        assert not any(i.code == "template-syntax" for i in issues)
