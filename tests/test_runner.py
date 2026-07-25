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
from seclab_taskflow_agent.results import ResultStore, ToolResult
from seclab_taskflow_agent.runner import (
    ResolvedModel,
    _aggregate_fanin,
    _build_prompts_to_run,
    _Branch,
    _capture_task_output,
    _completion,
    _fan_out_deploys,
    _merge_reusable_task,
    _resolve_model_config,
    _resolve_task_model,
    _resolve_task_models,
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
        with pytest.raises(ValueError, match="must contain exactly 1 task"):
            _merge_reusable_task(at, current)

    def test_raises_if_reusable_not_found(self):
        """ValueError raised when the reusable taskflow does not exist."""
        at = _mock_available_tools()
        at.get_taskflow.return_value = None

        current = TaskDefinition(uses="pkg.missing")
        with pytest.raises(ValueError, match="Failed to load reusable taskflow"):
            _merge_reusable_task(at, current)


# ===================================================================
# _resolve_task_model
# ===================================================================

class TestResolveTaskModel:
    """Tests for _resolve_task_model (pure function)."""

    def test_logical_name_mapped_to_provider_id(self):
        """A logical model name is resolved to the provider model ID."""
        model_id, _, _, _, _, _ = _resolve_task_model(
            TaskDefinition(model="fast"),
            model_keys=["fast"],
            model_dict={"fast": "gpt-4o-mini"},
            models_params={},
        )
        assert model_id == "gpt-4o-mini"

    def test_model_settings_from_config(self):
        """Settings from models_params are included in the result."""
        _, settings, _, _, _, _ = _resolve_task_model(
            TaskDefinition(model="fast"),
            model_keys=["fast"],
            model_dict={"fast": "gpt-4o-mini"},
            models_params={"fast": {"temperature": 0.7, "max_tokens": 100}},
        )
        assert settings["temperature"] == 0.7
        assert settings["max_tokens"] == 100

    def test_task_level_settings_override_config(self):
        """Task-level model_settings override config-level settings."""
        _, settings, _, _, _, _ = _resolve_task_model(
            TaskDefinition(model="fast", model_settings={"temperature": 0.2}),
            model_keys=["fast"],
            model_dict={"fast": "gpt-4o-mini"},
            models_params={"fast": {"temperature": 0.7, "max_tokens": 100}},
        )
        assert settings["temperature"] == 0.2
        assert settings["max_tokens"] == 100

    def test_engine_keys_extracted(self):
        """Engine keys (api_type, endpoint, token, backend) are popped from settings."""
        _, settings, api_type, endpoint, token, backend = _resolve_task_model(
            TaskDefinition(model="fast"),
            model_keys=["fast"],
            model_dict={"fast": "gpt-4o-mini"},
            models_params={
                "fast": {
                    "api_type": "responses",
                    "endpoint": "https://custom.api",
                    "token": "secret",
                    "backend": "anthropic_sdk",
                    "temperature": 0.5,
                }
            },
        )
        assert api_type == "responses"
        assert endpoint == "https://custom.api"
        assert token == "secret"  # noqa: S105
        assert backend == "anthropic_sdk"
        assert "api_type" not in settings
        assert "endpoint" not in settings
        assert "token" not in settings
        assert "backend" not in settings
        assert settings["temperature"] == 0.5

    def test_default_model_when_empty(self):
        """Empty model string falls back to DEFAULT_MODEL."""
        from seclab_taskflow_agent.agent import DEFAULT_MODEL

        model_id, _, _, _, _, _ = _resolve_task_model(
            TaskDefinition(model=""),
            model_keys=[],
            model_dict={},
            models_params={},
        )
        assert model_id == DEFAULT_MODEL

    def test_model_not_in_keys_passes_through(self):
        """A model name not in model_keys passes through as-is."""
        model_id, _, _, _, _, _ = _resolve_task_model(
            TaskDefinition(model="claude-3-opus"),
            model_keys=["fast", "smart"],
            model_dict={"fast": "gpt-4o-mini", "smart": "gpt-4o"},
            models_params={},
        )
        assert model_id == "claude-3-opus"

    def test_task_engine_keys_override_config(self):
        """Task-level model_settings can override engine keys from config."""
        _, _, api_type, endpoint, token, backend = _resolve_task_model(
            TaskDefinition(
                model="fast",
                model_settings={"api_type": "responses", "endpoint": "https://task.api", "backend": "anthropic_sdk"},
            ),
            model_keys=["fast"],
            model_dict={"fast": "gpt-4o-mini"},
            models_params={"fast": {"api_type": "chat_completions", "backend": "openai_agents"}},
        )
        assert api_type == "responses"
        assert endpoint == "https://task.api"
        assert backend == "anthropic_sdk"


# ===================================================================
# CLI model config override
# ===================================================================

class TestCliModelConfigOverride:
    """Tests for CLI model config overriding taskflow model_config_ref."""

    def test_cli_overrides_taskflow_model_config(self):
        """cli_model_config takes precedence over taskflow_doc.model_config_ref."""
        model_config_ref = self._resolve("taskflow.models.default", "cli.models.override")
        assert model_config_ref == "cli.models.override"

    def test_taskflow_model_config_used_when_cli_absent(self):
        """Taskflow model_config_ref is used when cli_model_config is None."""
        taskflow_ref = "taskflow.models.default"
        cli_ref = None

        model_config_ref = self._resolve(taskflow_ref, cli_ref)
        assert model_config_ref == taskflow_ref

    @staticmethod
    def _resolve(taskflow_ref: str, cli_ref: str | None) -> str:
        """Reproduce the override logic from run_main."""
        model_config_ref = taskflow_ref
        if cli_ref:
            model_config_ref = cli_ref
        return model_config_ref

    def test_cli_model_config_resolves_via_available_tools(self):
        """CLI-provided model config is resolved through _resolve_model_config."""
        at = _mock_available_tools()
        at.get_model_config.return_value = _make_model_config(
            models={"fast": "gpt-4o-mini"},
        )
        keys, mdict, params, api_type, backend = _resolve_model_config(at, "cli.override.ref")
        at.get_model_config.assert_called_once_with("cli.override.ref")
        assert mdict == {"fast": "gpt-4o-mini"}

    def test_cli_model_config_persisted_in_session(self):
        """cli_model_config is stored in session for deterministic resume."""
        from seclab_taskflow_agent.session import TaskflowSession

        session = TaskflowSession(
            taskflow_path="test.flow",
            cli_model_config="cli.models.fast",
        )
        assert session.cli_model_config == "cli.models.fast"

    def test_session_resume_restores_cli_model_config(self, tmp_path, monkeypatch):
        """Resumed session restores cli_model_config when not overridden."""
        monkeypatch.setattr("seclab_taskflow_agent.session.session_dir", lambda: tmp_path)
        from seclab_taskflow_agent.session import TaskflowSession

        session = TaskflowSession(
            taskflow_path="test.flow",
            cli_model_config="persisted.models.ref",
        )
        session.save()

        loaded = TaskflowSession.load(session.session_id)

        # Simulate the resume logic from run_main
        cli_model_config = None  # not passed on resume
        if not cli_model_config and loaded.cli_model_config:
            cli_model_config = loaded.cli_model_config

        assert cli_model_config == "persisted.models.ref"

    def test_session_resume_cli_override_takes_precedence(self, tmp_path, monkeypatch):
        """Explicit --model-config on resume overrides persisted value."""
        monkeypatch.setattr("seclab_taskflow_agent.session.session_dir", lambda: tmp_path)
        from seclab_taskflow_agent.session import TaskflowSession

        session = TaskflowSession(
            taskflow_path="test.flow",
            cli_model_config="persisted.models.ref",
        )
        session.save()

        loaded = TaskflowSession.load(session.session_id)

        # Simulate the resume logic from run_main with explicit override
        cli_model_config = "new.override.ref"
        if not cli_model_config and loaded.cli_model_config:
            cli_model_config = loaded.cli_model_config

        assert cli_model_config == "new.override.ref"

class TestBuildPromptsToRun:
    """Tests for _build_prompts_to_run (async, run via asyncio.run)."""

    @staticmethod
    def _store_with(*values: Any) -> ResultStore:
        """A ResultStore whose last recorded result carries JSON of *values[-1]*."""
        store = ResultStore()
        for v in values:
            store.record(ToolResult(text=json.dumps(v)))
        return store

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
                store=ResultStore(),
                available_tools=_mock_available_tools(),
                global_variables={},
                inputs={},
            )
        )
        assert result == ["hello world"]

    def test_repeat_with_json_array(self):
        """repeat_prompt with a JSON array generates one prompt per element."""
        items = [{"name": "apple"}, {"name": "banana"}]
        prompts = self._run(
            _build_prompts_to_run(
                task_prompt="Process {{ result.name }}",
                repeat_prompt=True,
                store=self._store_with(items),
                available_tools=_mock_available_tools(),
                global_variables={},
                inputs={},
            )
        )
        assert len(prompts) == 2
        assert "apple" in prompts[0]
        assert "banana" in prompts[1]

    def test_repeat_with_structured_result(self):
        """A structured (non-text) tool result is consumed directly."""
        store = ResultStore()
        store.record(ToolResult(structured=[1, 2, 3]))
        prompts = self._run(
            _build_prompts_to_run(
                task_prompt="n={{ result }}",
                repeat_prompt=True,
                store=store,
                available_tools=_mock_available_tools(),
                global_variables={},
                inputs={},
            )
        )
        assert len(prompts) == 3

    def test_repeat_with_empty_iterable(self):
        """repeat_prompt with an empty list renders no prompts."""
        prompts = self._run(
            _build_prompts_to_run(
                task_prompt="Process {{ result }}",
                repeat_prompt=True,
                store=self._store_with([]),
                available_tools=_mock_available_tools(),
                global_variables={},
                inputs={},
            )
        )
        assert prompts == []

    def test_repeat_over_empty_generator_emits_empty_notice(self):
        """An `over:` expression yielding an empty one-shot generator (e.g. a
        Jinja `map`/`select` filter) is detected as empty rather than silently
        producing zero prompts (a bare generator is always truthy)."""
        rendered: list[str] = []

        async def _capture(text: str, **_kw: Any) -> None:
            rendered.append(text)

        with patch("seclab_taskflow_agent.runner.render_model_output", _capture):
            prompts = asyncio.run(
                _build_prompts_to_run(
                    task_prompt="do {{ result }}",
                    repeat_prompt=True,
                    store=ResultStore(),
                    available_tools=_mock_available_tools(),
                    global_variables={},
                    inputs={},
                    outputs={"items": []},
                    over="outputs.items | map('upper')",
                )
            )
        assert prompts == []
        assert any("iterable is empty" in t for t in rendered)

    def test_raises_index_error_when_no_last_result(self):
        """IndexError when the store has no previous result."""
        with pytest.raises(IndexError):
            self._run(
                _build_prompts_to_run(
                    task_prompt="Process {{ result }}",
                    repeat_prompt=True,
                    store=ResultStore(),
                    available_tools=_mock_available_tools(),
                    global_variables={},
                    inputs={},
                )
            )

    def test_raises_value_error_on_non_json_result(self):
        """ValueError when the tool result text is not valid JSON."""
        store = ResultStore()
        store.record(ToolResult(text="not json!!"))
        with pytest.raises(ValueError, match="not valid JSON"):
            self._run(
                _build_prompts_to_run(
                    task_prompt="Process {{ result }}",
                    repeat_prompt=True,
                    store=store,
                    available_tools=_mock_available_tools(),
                    global_variables={},
                    inputs={},
                )
            )

    def test_pop_happens_after_successful_render(self):
        """The last result is consumed (popped) after all prompts render."""
        store = self._store_with([{"name": "x"}])
        self._run(
            _build_prompts_to_run(
                task_prompt="Process {{ result.name }}",
                repeat_prompt=True,
                store=store,
                available_tools=_mock_available_tools(),
                global_variables={},
                inputs={},
            )
        )
        assert store.last() is None

    def test_pop_does_not_happen_on_render_failure(self):
        """On template error the result is NOT consumed (available for retry)."""
        store = self._store_with([{"name": "x"}])
        with patch(
            "seclab_taskflow_agent.runner.render_template",
            side_effect=Exception("template boom"),
        ), pytest.raises(Exception, match="template boom"):
            self._run(
                _build_prompts_to_run(
                    task_prompt="Process {{ result.name }}",
                    repeat_prompt=True,
                    store=store,
                    available_tools=_mock_available_tools(),
                    global_variables={},
                    inputs={},
                )
            )
        assert store.last() is not None

    def test_raises_type_error_on_non_iterable_result(self):
        """TypeError when the result parses to a non-iterable (e.g. int)."""
        with pytest.raises(TypeError):
            self._run(
                _build_prompts_to_run(
                    task_prompt="Process {{ result }}",
                    repeat_prompt=True,
                    store=self._store_with(42),
                    available_tools=_mock_available_tools(),
                    global_variables={},
                    inputs={},
                )
            )

    def test_over_expression_selects_named_output(self):
        """An explicit ``over`` expression iterates a named output directly."""
        store = ResultStore()
        store.set_output("list_fns", {"functions": [{"name": "a"}, {"name": "b"}]})
        prompts = self._run(
            _build_prompts_to_run(
                task_prompt="fn={{ result.name }}",
                repeat_prompt=True,
                store=store,
                available_tools=_mock_available_tools(),
                global_variables={},
                inputs={},
                outputs=store.outputs,
                over="outputs.list_fns.functions",
            )
        )
        assert [p for p in prompts] == ["fn=a", "fn=b"]

    def test_over_does_not_consume_tool_result(self):
        """The explicit ``over`` path never pops the tool-result carry-over."""
        store = self._store_with(["ignored"])
        store.set_output("nums", [1, 2])
        self._run(
            _build_prompts_to_run(
                task_prompt="{{ result }}",
                repeat_prompt=True,
                store=store,
                available_tools=_mock_available_tools(),
                global_variables={},
                inputs={},
                outputs=store.outputs,
                over="outputs.nums",
            )
        )
        # the tool result is still present (over reads named data instead)
        assert store.last() is not None


# ===================================================================
# _resolve_task_models (multi-model resolution)
# ===================================================================

class TestResolveTaskModels:
    """Tests for _resolve_task_models (pure fan-out over model entries)."""

    def test_single_model_matches_resolve_task_model(self):
        """A singular-model task yields one ResolvedModel equal to the legacy path."""
        task = TaskDefinition(model="fast", model_settings={"temperature": 0.2})
        resolved = _resolve_task_models(
            task,
            model_keys=["fast"],
            model_dict={"fast": "gpt-4o-mini"},
            models_params={"fast": {"temperature": 0.7}},
        )
        assert len(resolved) == 1
        rm = resolved[0]
        legacy = _resolve_task_model(
            task, model_keys=["fast"], model_dict={"fast": "gpt-4o-mini"},
            models_params={"fast": {"temperature": 0.7}},
        )
        assert (rm.model, rm.model_settings, rm.api_type, rm.endpoint, rm.token, rm.backend) == legacy
        # Label falls back to the logical name the user wrote.
        assert rm.label == "fast"

    def test_default_model_label_when_empty(self):
        """An empty model resolves to DEFAULT_MODEL and labels with it."""
        from seclab_taskflow_agent.agent import DEFAULT_MODEL

        resolved = _resolve_task_models(
            TaskDefinition(user_prompt="hi"),
            model_keys=[], model_dict={}, models_params={},
        )
        assert len(resolved) == 1
        assert resolved[0].model == DEFAULT_MODEL
        assert resolved[0].label == DEFAULT_MODEL

    def test_fan_out_over_multiple_models(self):
        """Each ``models`` entry resolves to its own ResolvedModel."""
        task = TaskDefinition(user_prompt="hi", models=["fast", "smart"])
        resolved = _resolve_task_models(
            task,
            model_keys=["fast", "smart"],
            model_dict={"fast": "gpt-4o-mini", "smart": "gpt-4o"},
            models_params={},
        )
        assert [rm.model for rm in resolved] == ["gpt-4o-mini", "gpt-4o"]
        assert [rm.label for rm in resolved] == ["fast", "smart"]

    def test_per_entry_settings_and_engine_keys(self):
        """Per-entry model_settings override config and engine keys are extracted."""
        task = TaskDefinition(
            user_prompt="hi",
            models=[
                {"model": "fast", "model_settings": {"temperature": 0.1}},
                {"model": "native", "model_settings": {"backend": "anthropic_sdk", "api_type": "messages"}},
            ],
        )
        resolved = _resolve_task_models(
            task,
            model_keys=["fast", "native"],
            model_dict={"fast": "gpt-4o-mini", "native": "claude-opus-4.7"},
            models_params={"fast": {"temperature": 0.9, "top_p": 0.5}},
        )
        # entry-level temperature wins over config; config top_p preserved
        assert resolved[0].model_settings == {"temperature": 0.1, "top_p": 0.5}
        # engine keys are popped out of settings
        assert resolved[1].backend == "anthropic_sdk"
        assert resolved[1].api_type == "messages"
        assert "backend" not in resolved[1].model_settings
        assert "api_type" not in resolved[1].model_settings

    def test_unknown_model_passes_through(self):
        """A model name absent from model_keys passes through as the id."""
        task = TaskDefinition(user_prompt="hi", models=["claude-3-opus"])
        resolved = _resolve_task_models(
            task, model_keys=["fast"], model_dict={"fast": "gpt-4o-mini"}, models_params={},
        )
        assert resolved[0].model == "claude-3-opus"
        assert resolved[0].label == "claude-3-opus"


# ===================================================================
# _completion (task success reduction)
# ===================================================================

class TestCompletion:
    """Tests for the completion-policy reducer."""

    def test_empty_is_success(self):
        assert _completion([], "all") is True
        assert _completion([], "any") is True

    def test_all_policy(self):
        assert _completion([True, True], "all") is True
        assert _completion([True, False], "all") is False

    def test_any_policy(self):
        assert _completion([False, True], "any") is True
        assert _completion([False, False], "any") is False


# ===================================================================
# _fan_out_deploys (execution matrix + concurrency + completion)
# ===================================================================

def _rm(label: str) -> ResolvedModel:
    return ResolvedModel(
        model=label, model_settings={}, api_type="chat_completions",
        endpoint=None, token=None, backend=None, label=label,
    )


class TestFanOutDeploys:
    """Tests for the fan-out execution helper (no agent/MCP machinery)."""

    def test_runs_every_prompt_model_pair(self):
        """Deploy is invoked once for each (prompt, model) pair."""
        calls: list[tuple[str, str]] = []

        async def deploy(payload, rm):
            calls.append((payload, rm.label))
            return True

        items = [(p, rm) for p in ("p1", "p2") for rm in (_rm("m1"), _rm("m2"))]
        ok = asyncio.run(
            _fan_out_deploys(items, deploy, concurrent=True, concurrency=4, completion_policy="all")
        )
        assert ok is True
        assert set(calls) == {("p1", "m1"), ("p1", "m2"), ("p2", "m1"), ("p2", "m2")}

    def test_empty_work_items_is_success(self):
        async def deploy(payload, rm):  # pragma: no cover - never called
            raise AssertionError("deploy should not be called")

        ok = asyncio.run(
            _fan_out_deploys([], deploy, concurrent=True, concurrency=2, completion_policy="all")
        )
        assert ok is True

    def test_concurrency_bound_is_respected(self):
        """No more than ``concurrency`` deploys run simultaneously."""
        active = 0
        peak = 0

        async def deploy(payload, rm):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return True

        items = [("p", _rm(f"m{i}")) for i in range(6)]
        asyncio.run(
            _fan_out_deploys(items, deploy, concurrent=True, concurrency=2, completion_policy="all")
        )
        assert peak <= 2

    def test_sequential_path_preserves_order_and_propagates(self):
        """Non-concurrent path runs in order and lets exceptions propagate."""
        order: list[str] = []

        async def deploy(payload, rm):
            order.append(rm.label)
            if rm.label == "boom":
                raise RuntimeError("kaboom")
            return True

        items = [("p", _rm("a")), ("p", _rm("boom")), ("p", _rm("c"))]
        with pytest.raises(RuntimeError, match="kaboom"):
            asyncio.run(
                _fan_out_deploys(items, deploy, concurrent=False, concurrency=1, completion_policy="all")
            )
        # Stopped at the failing branch; "c" never ran.
        assert order == ["a", "boom"]

    def test_concurrent_exceptions_counted_as_failure(self):
        """In the concurrent path a raised branch becomes a failure, not a crash."""
        async def deploy(payload, rm):
            if rm.label == "bad":
                raise RuntimeError("branch failed")
            return True

        items = [("p", _rm("good")), ("p", _rm("bad"))]
        ok_all = asyncio.run(
            _fan_out_deploys(items, deploy, concurrent=True, concurrency=2, completion_policy="all")
        )
        ok_any = asyncio.run(
            _fan_out_deploys(items, deploy, concurrent=True, concurrency=2, completion_policy="any")
        )
        assert ok_all is False
        assert ok_any is True

    def test_completion_any_with_one_success(self):
        async def deploy(payload, rm):
            return rm.label == "winner"

        items = [("p", _rm("loser")), ("p", _rm("winner"))]
        assert asyncio.run(
            _fan_out_deploys(items, deploy, concurrent=True, concurrency=2, completion_policy="any")
        ) is True
        assert asyncio.run(
            _fan_out_deploys(items, deploy, concurrent=True, concurrency=2, completion_policy="all")
        ) is False


# ===================================================================
# _capture_task_output (typed named outputs)
# ===================================================================

class TestCaptureTaskOutput:
    """Tests for capturing a task's typed named output into the store."""

    def test_capture_schemaless_decodes_json(self):
        store = ResultStore()
        store.record(ToolResult(text=json.dumps({"functions": [1, 2]})))
        _capture_task_output(store, "list_fns", {}, "task-0", store.last())
        assert store.outputs["list_fns"] == {"functions": [1, 2]}

    def test_capture_schemaless_falls_back_to_text(self):
        store = ResultStore()
        store.record(ToolResult(text="a plain answer"))
        _capture_task_output(store, "answer", {}, "task-0", store.last())
        assert store.outputs["answer"] == "a plain answer"

    def test_capture_with_schema_validates(self):
        store = ResultStore()
        store.record(ToolResult(text=json.dumps({"functions": [{"name": "f", "body": "b"}]})))
        schema = {
            "type": "object",
            "properties": {
                "functions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "body": {"type": "string"}},
                        "required": ["name", "body"],
                    },
                }
            },
            "required": ["functions"],
        }
        _capture_task_output(store, "list_fns", schema, "task-0", store.last())
        assert store.outputs["list_fns"] == {"functions": [{"name": "f", "body": "b"}]}

    def test_capture_with_schema_validation_error(self):
        from jsonschema.exceptions import ValidationError

        store = ResultStore()
        store.record(ToolResult(text=json.dumps({"functions": "not-a-list"})))
        schema = {
            "type": "object",
            "properties": {"functions": {"type": "array"}},
            "required": ["functions"],
        }
        with pytest.raises(ValidationError):
            _capture_task_output(store, "list_fns", schema, "task-0", store.last())

    def test_capture_no_result_with_schema_raises(self):
        store = ResultStore()
        schema = {"type": "object", "properties": {"f": {"type": "string"}}, "required": ["f"]}
        with pytest.raises(ValueError, match="produced no tool result"):
            _capture_task_output(store, "x", schema, "task-0", None)

    def test_capture_no_result_schemaless_sets_none(self):
        store = ResultStore()
        _capture_task_output(store, "x", {}, "task-0", None)
        assert store.outputs["x"] is None

    def test_capture_prefers_structured(self):
        store = ResultStore()
        store.record(ToolResult(structured={"functions": ["a"]}))
        _capture_task_output(store, "out", {}, "task-0", store.last())
        assert store.outputs["out"] == {"functions": ["a"]}


# ===================================================================
# _aggregate_fanin (multi-model / cross-product fan-in)
# ===================================================================

def _branch(label: str, item: int, sink=None) -> _Branch:
    rm = _rm(label)
    b = _Branch(agents={}, prompt="p", rm=rm, item_index=item, label=label)
    for tr in sink or []:
        b.sink.append(tr)
    return b


class TestAggregateFanin:
    """Tests for the pure fan-in aggregator."""

    def test_empty_branches(self):
        assert _aggregate_fanin([]) == []

    def test_records_model_item_and_decoded_result(self):
        b = _branch("gpt_fast", 0, sink=[ToolResult(text=json.dumps({"score": 9}))])
        records = _aggregate_fanin([b])
        assert records == [{"model": "gpt_fast", "item": 0, "result": {"score": 9}}]

    def test_uses_last_result_of_branch(self):
        b = _branch("m", 1, sink=[ToolResult(text="1"), ToolResult(text="2")])
        assert _aggregate_fanin([b])[0]["result"] == 2

    def test_non_json_text_falls_back_to_text(self):
        b = _branch("m", 0, sink=[ToolResult(text="not json")])
        assert _aggregate_fanin([b])[0]["result"] == "not json"

    def test_structured_result_preferred(self):
        b = _branch("m", 0, sink=[ToolResult(structured=[1, 2, 3])])
        assert _aggregate_fanin([b])[0]["result"] == [1, 2, 3]

    def test_empty_sink_yields_none(self):
        b = _branch("m", 2, sink=[])
        assert _aggregate_fanin([b])[0]["result"] is None

    def test_multiple_branches_preserve_order(self):
        branches = [
            _branch("m1", 0, sink=[ToolResult(text="10")]),
            _branch("m2", 0, sink=[ToolResult(text="20")]),
            _branch("m1", 1, sink=[ToolResult(text="30")]),
        ]
        records = _aggregate_fanin(branches)
        assert [(r["model"], r["item"], r["result"]) for r in records] == [
            ("m1", 0, 10),
            ("m2", 0, 20),
            ("m1", 1, 30),
        ]
