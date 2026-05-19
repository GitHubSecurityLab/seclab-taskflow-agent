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
    TaskDefinition,
    TaskflowDocument,
    TaskflowHeader,
    TaskWrapper,
)
from seclab_taskflow_agent.runner import (
    _build_prompts_to_run,
    _merge_reusable_task,
    _resolve_model_config,
    _resolve_task_model,
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
        keys, mdict, params, api_type, backend = _resolve_model_config(at, "ref")
        assert set(keys) == {"fast", "smart"}
        assert mdict == {"fast": "gpt-4o-mini", "smart": "gpt-4o"}
        assert params == {}
        assert api_type == "chat_completions"
        assert backend is None

    def test_api_type_flows_through(self):
        """api_type from the config document is returned."""
        at = _mock_available_tools()
        at.get_model_config.return_value = _make_model_config(
            models={"m1": "provider-model"},
            api_type="responses",
        )
        _, _, _, api_type, _ = _resolve_model_config(at, "ref")
        assert api_type == "responses"

    def test_model_settings_extraction(self):
        """Per-model settings are returned and keyed by logical name."""
        at = _mock_available_tools()
        at.get_model_config.return_value = _make_model_config(
            models={"m1": "provider-m1"},
            model_settings={"m1": {"temperature": 0.5}},
        )
        _, _, params, _, _ = _resolve_model_config(at, "ref")
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
