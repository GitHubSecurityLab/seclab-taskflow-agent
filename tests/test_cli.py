# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the Typer CLI module."""

from __future__ import annotations

import pytest
import typer

from seclab_taskflow_agent.cli import _parse_global


class TestParseGlobal:
    """Tests for _parse_global KEY=VALUE parsing."""

    def test_valid_key_value(self):
        """Standard KEY=VALUE is parsed correctly."""
        assert _parse_global("fruit=apple") == ("fruit", "apple")

    def test_missing_equals_raises(self):
        """A string without '=' raises BadParameter."""
        with pytest.raises(typer.BadParameter, match="Expected KEY=VALUE"):
            _parse_global("no_equals_here")

    def test_value_with_equals_sign(self):
        """Only the first '=' is used as the delimiter."""
        key, val = _parse_global("url=https://example.com?foo=bar")
        assert key == "url"
        assert val == "https://example.com?foo=bar"

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace in key and value is stripped."""
        key, val = _parse_global("  key  =  value  ")
        assert key == "key"
        assert val == "value"

    def test_empty_value(self):
        """An empty value after '=' is allowed."""
        key, val = _parse_global("key=")
        assert key == "key"
        assert val == ""

    def test_empty_key(self):
        """An empty key before '=' is technically allowed by the parser."""
        key, val = _parse_global("=value")
        assert key == ""
        assert val == "value"


class TestDebugEnvParsing:
    """Tests for the TASK_AGENT_DEBUG environment variable expression."""

    @staticmethod
    def _is_debug(env_value: str) -> bool:
        """Reproduce the debug expression from cli.py."""
        return env_value.strip().lower() in ("1", "true", "yes")

    def test_zero_is_false(self):
        assert self._is_debug("0") is False

    def test_one_is_true(self):
        assert self._is_debug("1") is True

    def test_true_string_is_true(self):
        assert self._is_debug("true") is True

    def test_TRUE_string_is_true(self):
        assert self._is_debug("TRUE") is True

    def test_yes_string_is_true(self):
        assert self._is_debug("yes") is True

    def test_empty_string_is_false(self):
        assert self._is_debug("") is False

    def test_false_string_is_false(self):
        assert self._is_debug("false") is False

    def test_whitespace_trimmed(self):
        assert self._is_debug("  1  ") is True

    def test_random_text_is_false(self):
        assert self._is_debug("enabled") is False


class TestLintAndSchema:
    """Tests for the --lint and --schema CLI helpers."""

    def test_run_lint_returns_zero_when_clean(self, monkeypatch):
        from seclab_taskflow_agent import cli

        def _no_issues(*_a, **_k):
            return []

        monkeypatch.setattr("seclab_taskflow_agent.linting.lint_taskflow", _no_issues)
        assert cli._run_lint(object(), "pkg.flow", strict=False, cli_model_config=None) == 0

    def test_run_lint_returns_one_on_error(self, monkeypatch):
        from seclab_taskflow_agent import cli
        from seclab_taskflow_agent.linting import LintIssue

        def _error(*_a, **_k):
            return [LintIssue("error", "x", "boom")]

        monkeypatch.setattr("seclab_taskflow_agent.linting.lint_taskflow", _error)
        assert cli._run_lint(object(), "pkg.flow", strict=False, cli_model_config=None) == 1

    def test_run_lint_warnings_only_returns_zero(self, monkeypatch):
        from seclab_taskflow_agent import cli
        from seclab_taskflow_agent.linting import LintIssue

        def _warn(*_a, **_k):
            return [LintIssue("warning", "y", "meh")]

        monkeypatch.setattr("seclab_taskflow_agent.linting.lint_taskflow", _warn)
        assert cli._run_lint(object(), "pkg.flow", strict=False, cli_model_config=None) == 0

    def test_run_lint_integration_on_example(self):
        from seclab_taskflow_agent import cli
        from seclab_taskflow_agent.available_tools import AvailableTools

        # A shipped example must lint without errors (exit 0).
        assert cli._run_lint(AvailableTools(), "examples.taskflows.echo", strict=False, cli_model_config=None) == 0

    def test_dump_schemas_outputs_all_document_types(self, capsys):
        import json

        from seclab_taskflow_agent import cli

        cli._dump_schemas()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert set(data.keys()) == {"taskflow", "personality", "toolbox", "model_config", "prompt"}
