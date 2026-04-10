# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for runner helper functions (no API calls)."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seclab_taskflow_agent.models import (
    ModelConfigDocument,
    PersonalityDocument,
    TaskDefinition,
    TaskflowDocument,
    TaskflowHeader,
    TaskWrapper,
)
from seclab_taskflow_agent.runner import (
    LoopDetectedError,
    _build_prompts_to_run,
    _merge_reusable_task,
    _resolve_model_config,
    _resolve_task_model,
    check_consecutive_tool_loop,
    read_tool_log,
    run_main,
    write_auto_save,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_header() -> TaskflowHeader:
    return TaskflowHeader(version="1.0", filetype="taskflow")


def _make_model_config_header() -> TaskflowHeader:
    return TaskflowHeader(version="1.0", filetype="model_config")


def _make_model_config(
    models: dict[str, str] | None = None,
    model_settings: dict[str, dict[str, Any]] | None = None,
    api_type: str = "chat_completions",
) -> ModelConfigDocument:
    return ModelConfigDocument(
        **{
            "seclab-taskflow-agent": _make_model_config_header(),
            "api_type": api_type,
            "models": models or {},
            "model_settings": model_settings or {},
        }
    )


def _make_taskflow_doc(tasks: list[TaskDefinition]) -> TaskflowDocument:
    return TaskflowDocument(
        **{
            "seclab-taskflow-agent": _make_header(),
            "taskflow": [TaskWrapper(task=t) for t in tasks],
        }
    )


def _make_personality() -> PersonalityDocument:
    return PersonalityDocument(
        **{
            "seclab-taskflow-agent": TaskflowHeader(version="1.0", filetype="personality"),
            "personality": "test bot",
            "task": "do things",
            "toolboxes": [],
        }
    )


def _mock_available_tools() -> MagicMock:
    return MagicMock()


# ===================================================================
# _resolve_model_config
# ===================================================================

class TestResolveModelConfig:
    """Tests for _resolve_model_config."""

    def test_basic_model_resolution(self):
        """Model keys and dict are extracted from config."""
        at = _mock_available_tools()
        at.get_model_config.return_value = _make_model_config(
            models={"fast": "gpt-4o-mini", "smart": "gpt-4o"},
        )
        keys, mdict, params, api_type = _resolve_model_config(at, "ref")
        assert set(keys) == {"fast", "smart"}
        assert mdict == {"fast": "gpt-4o-mini", "smart": "gpt-4o"}
        assert params == {}
        assert api_type == "chat_completions"

    def test_api_type_flows_through(self):
        """api_type from the config document is returned."""
        at = _mock_available_tools()
        at.get_model_config.return_value = _make_model_config(
            models={"m1": "provider-model"},
            api_type="responses",
        )
        _, _, _, api_type = _resolve_model_config(at, "ref")
        assert api_type == "responses"

    def test_model_settings_extraction(self):
        """Per-model settings are returned and keyed by logical name."""
        at = _mock_available_tools()
        at.get_model_config.return_value = _make_model_config(
            models={"m1": "provider-m1"},
            model_settings={"m1": {"temperature": 0.5}},
        )
        _, _, params, _ = _resolve_model_config(at, "ref")
        assert params == {"m1": {"temperature": 0.5}}

    def test_validation_error_on_non_dict_settings(self):
        """Pydantic rejects non-dict model_settings at parse time."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _make_model_config(models={"m1": "p-m1"}, model_settings="not-a-dict")


# ===================================================================
# _merge_reusable_task
# ===================================================================

class TestMergeReusableTask:
    """Tests for _merge_reusable_task."""

    def test_current_fields_override_parent(self):
        """Fields explicitly set on the current task override the parent."""
        parent = TaskDefinition(name="parent", user_prompt="parent prompt", model="slow")
        doc = _make_taskflow_doc([parent])

        at = _mock_available_tools()
        at.get_taskflow.return_value = doc

        current = TaskDefinition(uses="pkg.reusable", name="child", model="fast")
        merged = _merge_reusable_task(at, current)
        assert merged.name == "child"
        assert merged.model == "fast"
        # Parent's prompt should fill in where child uses the default
        assert merged.user_prompt == "parent prompt"

    def test_parent_defaults_fill_in(self):
        """Parent defaults are used when the current task does not set a field."""
        parent = TaskDefinition(
            name="parent",
            user_prompt="do something",
            headless=True,
            must_complete=True,
        )
        doc = _make_taskflow_doc([parent])

        at = _mock_available_tools()
        at.get_taskflow.return_value = doc

        current = TaskDefinition(uses="pkg.reusable", name="override-name")
        merged = _merge_reusable_task(at, current)
        assert merged.name == "override-name"
        assert merged.headless is True
        assert merged.must_complete is True

    def test_raises_if_reusable_has_multiple_tasks(self):
        """ValueError raised when reusable taskflow has more than 1 task."""
        t1 = TaskDefinition(name="t1")
        t2 = TaskDefinition(name="t2")
        doc = _make_taskflow_doc([t1, t2])

        at = _mock_available_tools()
        at.get_taskflow.return_value = doc

        current = TaskDefinition(uses="pkg.multi")
        with pytest.raises(ValueError, match="only contain 1 task"):
            _merge_reusable_task(at, current)

    def test_raises_if_reusable_not_found(self):
        """ValueError raised when the reusable taskflow does not exist."""
        at = _mock_available_tools()
        at.get_taskflow.return_value = None

        current = TaskDefinition(uses="pkg.missing")
        with pytest.raises(ValueError, match="No such reusable taskflow"):
            _merge_reusable_task(at, current)


# ===================================================================
# _resolve_task_model
# ===================================================================

class TestResolveTaskModel:
    """Tests for _resolve_task_model (pure function)."""

    def test_logical_name_mapped_to_provider_id(self):
        """A logical model name is resolved to the provider model ID."""
        model_id, _, _, _, _ = _resolve_task_model(
            TaskDefinition(model="fast"),
            model_keys=["fast"],
            model_dict={"fast": "gpt-4o-mini"},
            models_params={},
        )
        assert model_id == "gpt-4o-mini"

    def test_model_settings_from_config(self):
        """Settings from models_params are included in the result."""
        _, settings, _, _, _ = _resolve_task_model(
            TaskDefinition(model="fast"),
            model_keys=["fast"],
            model_dict={"fast": "gpt-4o-mini"},
            models_params={"fast": {"temperature": 0.7, "max_tokens": 100}},
        )
        assert settings["temperature"] == 0.7
        assert settings["max_tokens"] == 100

    def test_task_level_settings_override_config(self):
        """Task-level model_settings override config-level settings."""
        _, settings, _, _, _ = _resolve_task_model(
            TaskDefinition(model="fast", model_settings={"temperature": 0.2}),
            model_keys=["fast"],
            model_dict={"fast": "gpt-4o-mini"},
            models_params={"fast": {"temperature": 0.7, "max_tokens": 100}},
        )
        assert settings["temperature"] == 0.2
        assert settings["max_tokens"] == 100

    def test_engine_keys_extracted(self):
        """Engine keys (api_type, endpoint, token) are popped from settings."""
        _, settings, api_type, endpoint, token = _resolve_task_model(
            TaskDefinition(model="fast"),
            model_keys=["fast"],
            model_dict={"fast": "gpt-4o-mini"},
            models_params={
                "fast": {
                    "api_type": "responses",
                    "endpoint": "https://custom.api",
                    "token": "secret",
                    "temperature": 0.5,
                }
            },
        )
        assert api_type == "responses"
        assert endpoint == "https://custom.api"
        assert token == "secret"  # noqa: S105
        assert "api_type" not in settings
        assert "endpoint" not in settings
        assert "token" not in settings
        assert settings["temperature"] == 0.5

    def test_default_model_when_empty(self):
        """Empty model string falls back to DEFAULT_MODEL."""
        from seclab_taskflow_agent.agent import DEFAULT_MODEL

        model_id, _, _, _, _ = _resolve_task_model(
            TaskDefinition(model=""),
            model_keys=[],
            model_dict={},
            models_params={},
        )
        assert model_id == DEFAULT_MODEL

    def test_model_not_in_keys_passes_through(self):
        """A model name not in model_keys passes through as-is."""
        model_id, _, _, _, _ = _resolve_task_model(
            TaskDefinition(model="claude-3-opus"),
            model_keys=["fast", "smart"],
            model_dict={"fast": "gpt-4o-mini", "smart": "gpt-4o"},
            models_params={},
        )
        assert model_id == "claude-3-opus"

    def test_task_engine_keys_override_config(self):
        """Task-level model_settings can override engine keys from config."""
        _, _, api_type, endpoint, token = _resolve_task_model(
            TaskDefinition(
                model="fast",
                model_settings={"api_type": "responses", "endpoint": "https://task.api"},
            ),
            model_keys=["fast"],
            model_dict={"fast": "gpt-4o-mini"},
            models_params={"fast": {"api_type": "chat_completions"}},
        )
        assert api_type == "responses"
        assert endpoint == "https://task.api"


# ===================================================================
# _build_prompts_to_run
# ===================================================================

class TestBuildPromptsToRun:
    """Tests for _build_prompts_to_run (async, run via asyncio.run)."""

    @staticmethod
    def _result_entry(data: Any) -> str:
        """Build a JSON string mimicking an MCP tool result."""
        return json.dumps({"text": json.dumps(data)})

    @staticmethod
    def _run(coro):
        """Run an async coroutine with render_model_output mocked out."""
        with patch("seclab_taskflow_agent.runner.render_model_output", new_callable=AsyncMock):
            return asyncio.run(coro)

    def test_non_repeat_returns_single_prompt(self):
        """Without repeat_prompt, the original prompt is returned as-is."""
        result = self._run(
            _build_prompts_to_run(
                task_prompt="hello world",
                repeat_prompt=False,
                last_mcp_tool_results=[],
                available_tools=_mock_available_tools(),
                global_variables={},
                inputs={},
            )
        )
        assert result == ["hello world"]

    def test_repeat_with_json_array(self):
        """repeat_prompt with a JSON array generates one prompt per element."""
        items = [{"name": "apple"}, {"name": "banana"}]
        results = [self._result_entry(items)]
        prompts = self._run(
            _build_prompts_to_run(
                task_prompt="Process {{ result.name }}",
                repeat_prompt=True,
                last_mcp_tool_results=results,
                available_tools=_mock_available_tools(),
                global_variables={},
                inputs={},
            )
        )
        assert len(prompts) == 2
        assert "apple" in prompts[0]
        assert "banana" in prompts[1]

    def test_repeat_with_dict_items(self):
        """repeat_prompt iterates over dict keys when result is a dict."""
        data = {"a": 1, "b": 2}
        results = [self._result_entry(data)]
        prompts = self._run(
            _build_prompts_to_run(
                task_prompt="Key: {{ result }}",
                repeat_prompt=True,
                last_mcp_tool_results=results,
                available_tools=_mock_available_tools(),
                global_variables={},
                inputs={},
            )
        )
        assert len(prompts) == 2

    def test_repeat_with_empty_iterable(self):
        """repeat_prompt with an empty list renders no prompts."""
        results = [self._result_entry([])]
        prompts = self._run(
            _build_prompts_to_run(
                task_prompt="Process {{ result }}",
                repeat_prompt=True,
                last_mcp_tool_results=results,
                available_tools=_mock_available_tools(),
                global_variables={},
                inputs={},
            )
        )
        assert prompts == []

    def test_raises_index_error_when_no_last_result(self):
        """IndexError when last_mcp_tool_results is empty."""
        with pytest.raises(IndexError):
            self._run(
                _build_prompts_to_run(
                    task_prompt="Process {{ result }}",
                    repeat_prompt=True,
                    last_mcp_tool_results=[],
                    available_tools=_mock_available_tools(),
                    global_variables={},
                    inputs={},
                )
            )

    def test_raises_value_error_on_non_json_result(self):
        """ValueError when MCP result text is not valid JSON."""
        results = [json.dumps({"text": "not json!!"})]
        with pytest.raises(ValueError, match="not valid JSON"):
            self._run(
                _build_prompts_to_run(
                    task_prompt="Process {{ result }}",
                    repeat_prompt=True,
                    last_mcp_tool_results=results,
                    available_tools=_mock_available_tools(),
                    global_variables={},
                    inputs={},
                )
            )

    def test_pop_happens_after_successful_render(self):
        """The last result is only consumed after all prompts render."""
        items = [{"name": "x"}]
        results = [self._result_entry(items)]
        original_len = len(results)

        self._run(
            _build_prompts_to_run(
                task_prompt="Process {{ result.name }}",
                repeat_prompt=True,
                last_mcp_tool_results=results,
                available_tools=_mock_available_tools(),
                global_variables={},
                inputs={},
            )
        )
        # After success, the entry should be consumed
        assert len(results) == original_len - 1

    def test_pop_does_not_happen_on_render_failure(self):
        """On template error the result is NOT consumed (available for retry)."""
        items = [{"name": "x"}]
        results = [self._result_entry(items)]

        with patch(
            "seclab_taskflow_agent.runner.render_template",
            side_effect=Exception("template boom"),
        ), pytest.raises(Exception, match="template boom"):
            self._run(
                _build_prompts_to_run(
                    task_prompt="Process {{ result.name }}",
                    repeat_prompt=True,
                    last_mcp_tool_results=results,
                    available_tools=_mock_available_tools(),
                    global_variables={},
                    inputs={},
                )
            )
        # Result should still be there for retry
        assert len(results) == 1

    def test_raises_type_error_on_non_iterable_result(self):
        """TypeError when MCP result parses to a non-iterable (e.g. int)."""
        results = [self._result_entry(42)]
        with pytest.raises(TypeError):
            self._run(
                _build_prompts_to_run(
                    task_prompt="Process {{ result }}",
                    repeat_prompt=True,
                    last_mcp_tool_results=results,
                    available_tools=_mock_available_tools(),
                    global_variables={},
                    inputs={},
                )
            )


# ===================================================================
# Auto-save scaffolding
# ===================================================================


class TestAutoSave:
    """Tests for write_auto_save / read_tool_log (module-level functions)."""

    def test_disabled_when_no_dir(self):
        """read_tool_log returns [] when dir is empty string."""
        assert read_tool_log("") == []

    def test_write_then_read_roundtrip(self, tmp_path):
        """write_auto_save produces entries that read_tool_log can read back."""
        d = str(tmp_path)
        write_auto_save(d, turn=1, tool_name="search_code", result="found 5")
        write_auto_save(d, turn=2, tool_name="read_file", result="contents")
        entries = read_tool_log(d)
        assert len(entries) == 2
        assert entries[0]["turn"] == 1
        assert entries[0]["tool"] == "search_code"
        assert entries[1]["turn"] == 2

    def test_log_format_has_turn_tool_preview(self, tmp_path):
        """Each NDJSON entry has the expected keys."""
        d = str(tmp_path)
        write_auto_save(d, turn=7, tool_name="search_code", result="found 5 matches")
        entries = read_tool_log(d)
        assert len(entries) == 1
        assert entries[0] == {"turn": 7, "tool": "search_code", "result_preview": "found 5 matches"}

    def test_result_truncated_to_2000(self, tmp_path):
        """Result preview is capped at 2000 characters."""
        d = str(tmp_path)
        write_auto_save(d, turn=1, tool_name="big", result="x" * 5000)
        entries = read_tool_log(d)
        assert len(entries[0]["result_preview"]) == 2000

    def test_survives_write_failure(self):
        """write_auto_save suppresses write errors without crashing."""
        with patch("builtins.open", side_effect=OSError("disk full")):
            write_auto_save("/tmp/any", turn=1, tool_name="t", result="r")

    def test_read_skips_corrupt_trailing_line(self, tmp_path):
        """read_tool_log skips truncated/corrupt lines without discarding valid ones."""
        import os

        d = str(tmp_path)
        write_auto_save(d, turn=1, tool_name="good", result="ok")
        # Append a corrupt line simulating crash mid-write
        log_path = os.path.join(d, "auto_save_tool_log.ndjson")
        with open(log_path, "a") as f:
            f.write('{"truncated\n')
        entries = read_tool_log(d)
        assert len(entries) == 1
        assert entries[0]["tool"] == "good"

    def test_read_empty_dir(self, tmp_path):
        """read_tool_log on a dir with no log file returns []."""
        assert read_tool_log(str(tmp_path)) == []


# ===================================================================
# Loop detection
# ===================================================================


class TestLoopDetection:
    """Tests for check_consecutive_tool_loop (real implementation)."""

    def test_raises_after_threshold(self):
        """Reaching the threshold raises LoopDetectedError with correct attrs."""
        name, count = [""], [0]
        for _ in range(4):
            check_consecutive_tool_loop("search_code", name, count, threshold=5)
        with pytest.raises(LoopDetectedError) as exc_info:
            check_consecutive_tool_loop("search_code", name, count, threshold=5)
        assert exc_info.value.tool_name == "search_code"
        assert exc_info.value.count == 5

    def test_different_tools_reset_counter(self):
        """Alternating tools never triggers — counter resets on name change."""
        name, count = [""], [0]
        for _ in range(20):
            check_consecutive_tool_loop("tool_a", name, count, threshold=3)
            check_consecutive_tool_loop("tool_b", name, count, threshold=3)

    def test_no_raise_when_disabled(self):
        """Threshold 0 disables detection entirely."""
        name, count = [""], [0]
        for _ in range(100):
            check_consecutive_tool_loop("same_tool", name, count, threshold=0)

    def test_negative_threshold_disables(self):
        """Negative threshold also disables detection."""
        name, count = [""], [0]
        for _ in range(100):
            check_consecutive_tool_loop("same_tool", name, count, threshold=-1)

    def test_counter_resets_on_different_tool(self):
        """Inserting a different tool resets the streak."""
        name, count = [""], [0]
        for _ in range(4):
            check_consecutive_tool_loop("search_code", name, count, threshold=5)
        assert count[0] == 4
        check_consecutive_tool_loop("read_file", name, count, threshold=5)
        assert count[0] == 1
        assert name[0] == "read_file"

    def test_state_mutation_visible_to_caller(self):
        """The function mutates the lists in-place so callers see updates."""
        name, count = [""], [0]
        check_consecutive_tool_loop("tool_x", name, count, threshold=10)
        assert name[0] == "tool_x"
        assert count[0] == 1
        check_consecutive_tool_loop("tool_x", name, count, threshold=10)
        assert count[0] == 2

    def test_task_definition_accepts_max_consecutive_same_tool(self):
        t = TaskDefinition(max_consecutive_same_tool=10)
        assert t.max_consecutive_same_tool == 10

    def test_task_definition_defaults_to_none(self):
        t = TaskDefinition()
        assert t.max_consecutive_same_tool is None

    def test_explicit_zero_disables(self):
        t = TaskDefinition(max_consecutive_same_tool=0)
        assert t.max_consecutive_same_tool == 0

    def test_existing_yaml_without_field_parses(self):
        """TaskDefinition without max_consecutive_same_tool parses fine."""
        t = TaskDefinition(name="legacy", agents=["p.a"], user_prompt="Hello")
        assert t.max_consecutive_same_tool is None


# ===================================================================
# Loop detection integration (drives run_main → on_tool_end_hook)
# ===================================================================


def _run_main_with_loop(
    monkeypatch,
    tmp_path,
    task_kwargs,
    n_tool_calls,
    tool_name="search_code",
):
    """Helper: run run_main with a fake deploy that fires N same-tool completions.

    Returns the LoopDetectedError if raised, or None if run_main completed.
    """
    task = TaskDefinition(
        agents=["test.personality"],
        user_prompt="do stuff",
        **task_kwargs,
    )
    doc = _make_taskflow_doc([task])

    at = _mock_available_tools()
    at.get_taskflow.return_value = doc
    at.get_personality.return_value = _make_personality()

    async def fake_deploy(_at, _agents, _prompt, **kwargs):
        run_hooks = kwargs.get("run_hooks")
        if run_hooks and run_hooks.on_tool_end:
            ctx = MagicMock()
            agent = MagicMock()
            tool = MagicMock()
            tool.name = tool_name
            for _ in range(n_tool_calls):
                await run_hooks.on_tool_end(ctx, agent, tool, "result")
        return True

    monkeypatch.setattr("seclab_taskflow_agent.runner.deploy_task_agents", fake_deploy)
    monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)

    with patch("seclab_taskflow_agent.runner.render_model_output", new_callable=AsyncMock):
            try:
                asyncio.run(run_main(at, None, "test.taskflow", {}, None))
            except LoopDetectedError as exc:
                return exc
    return None


class TestLoopDetectionIntegration:
    """Integration tests that drive run_main → on_tool_end_hook."""

    def test_triggers_via_task_field(self, monkeypatch, tmp_path):
        """max_consecutive_same_tool=3 on the task fires after 3 calls."""
        exc = _run_main_with_loop(
            monkeypatch, tmp_path,
            task_kwargs={"max_consecutive_same_tool": 3},
            n_tool_calls=5,
        )
        assert exc is not None
        assert exc.tool_name == "search_code"
        assert exc.count == 3

    def test_triggers_via_env_fallback(self, monkeypatch, tmp_path):
        """LOOP_MAX_CONSECUTIVE env var is used when task field is None."""
        monkeypatch.setenv("LOOP_MAX_CONSECUTIVE", "4")
        exc = _run_main_with_loop(
            monkeypatch, tmp_path,
            task_kwargs={},  # no task-level field → env fallback
            n_tool_calls=6,
        )
        assert exc is not None
        assert exc.count == 4

    def test_disabled_when_zero(self, monkeypatch, tmp_path):
        """max_consecutive_same_tool=0 disables even with env set."""
        monkeypatch.setenv("LOOP_MAX_CONSECUTIVE", "3")
        exc = _run_main_with_loop(
            monkeypatch, tmp_path,
            task_kwargs={"max_consecutive_same_tool": 0},
            n_tool_calls=10,
        )
        assert exc is None

    def test_async_task_bypass(self, monkeypatch, tmp_path):
        """Loop detection is skipped for async tasks."""
        task = TaskDefinition(
            agents=["test.personality"],
            user_prompt="do stuff",
            max_consecutive_same_tool=3,
            **{"async": True},
        )
        doc = _make_taskflow_doc([task])

        at = _mock_available_tools()
        at.get_taskflow.return_value = doc
        at.get_personality.return_value = _make_personality()

        async def fake_deploy(_at, _agents, _prompt, **kwargs):
            run_hooks = kwargs.get("run_hooks")
            if run_hooks and run_hooks.on_tool_end:
                ctx, agent, tool = MagicMock(), MagicMock(), MagicMock()
                tool.name = "search_code"
                for _ in range(10):
                    await run_hooks.on_tool_end(ctx, agent, tool, "r")
            return True

        monkeypatch.setattr("seclab_taskflow_agent.runner.deploy_task_agents", fake_deploy)
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)

        with patch("seclab_taskflow_agent.runner.render_model_output", new_callable=AsyncMock):
            # Should NOT raise — async bypass
            asyncio.run(run_main(at, None, "test.taskflow", {}, None))
