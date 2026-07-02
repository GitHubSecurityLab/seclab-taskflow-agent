# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the CLI command routing / validation (via Typer's CliRunner).

Only the paths that return early (no model calls, no os._exit) are exercised:
--schema, --lint, --manifest, and the mutual-exclusivity / usage errors.
Error messages are written to stderr, so error cases assert on the exit code.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from seclab_taskflow_agent.cli import app

runner = CliRunner()


class TestCliRouting:
    def test_schema_prints_all_document_types(self):
        result = runner.invoke(app, ["--schema"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert set(data.keys()) == {"taskflow", "personality", "toolbox", "model_config", "prompt"}

    def test_lint_valid_example_exits_zero(self):
        result = runner.invoke(app, ["--lint", "-t", "examples.taskflows.echo"])
        assert result.exit_code == 0
        assert "No issues found" in result.stdout

    def test_lint_requires_taskflow(self):
        result = runner.invoke(app, ["--lint"])
        assert result.exit_code == 1

    def test_lint_bad_taskflow_exits_one(self):
        result = runner.invoke(app, ["--lint", "-t", "examples.taskflows.does_not_exist"])
        assert result.exit_code == 1

    def test_manifest_missing_session_exits_one(self):
        result = runner.invoke(app, ["--manifest", "no-such-session-id"])
        assert result.exit_code == 1

    def test_mutually_exclusive_p_and_t(self):
        result = runner.invoke(app, ["-p", "a.b", "-t", "c.d"])
        assert result.exit_code == 1

    def test_resume_not_combinable(self):
        result = runner.invoke(app, ["--resume", "abc", "-t", "c.d"])
        assert result.exit_code == 1
