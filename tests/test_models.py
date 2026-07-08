# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for Pydantic grammar models."""

import pytest
from pydantic import ValidationError

from seclab_taskflow_agent.models import (
    ModelConfigDocument,
    ModelEntry,
    PersonalityDocument,
    PromptDocument,
    ServerParams,
    TaskDefinition,
    TaskflowDocument,
    TaskflowHeader,
    ToolboxDocument,
)


class TestTaskflowHeader:
    """Test the grammar header validation."""

    def test_string_version(self):
        h = TaskflowHeader(version="1.0", filetype="taskflow")
        assert h.version == "1.0"

    def test_integer_version_normalised(self):
        h = TaskflowHeader(version=1, filetype="taskflow")
        assert h.version == "1.0"

    def test_float_version_normalised(self):
        h = TaskflowHeader(version=1.0, filetype="taskflow")
        assert h.version == "1.0"

    def test_unsupported_version_rejected(self):
        with pytest.raises(ValidationError, match="Unsupported version"):
            TaskflowHeader(version="2.0", filetype="taskflow")

    def test_filetype_preserved(self):
        h = TaskflowHeader(version="1.0", filetype="personality")
        assert h.filetype == "personality"


class TestTaskDefinition:
    """Test single task validation."""

    def test_defaults(self):
        t = TaskDefinition()
        assert t.agents == []
        assert t.user_prompt == ""
        assert t.must_complete is False
        assert t.async_task is False
        assert t.async_limit == 5
        assert t.max_steps == 0

    def test_all_fields(self):
        t = TaskDefinition(
            name="test-task",
            agents=["personality.a"],
            user_prompt="Hello {{ globals.x }}",
            model="gpt-4o",
            must_complete=True,
            headless=True,
            repeat_prompt=True,
            toolboxes=["toolbox.a"],
            env={"KEY": "val"},
            max_steps=20,
            **{"async": True},
            async_limit=3,
        )
        assert t.name == "test-task"
        assert t.async_task is True
        assert t.async_limit == 3
        assert t.max_steps == 20

    def test_run_and_prompt_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            TaskDefinition(run="echo hi", user_prompt="Hello")

    def test_capture_defaults_to_tool_result(self):
        assert TaskDefinition(user_prompt="hi").capture == "tool_result"

    def test_capture_response_rejects_shell_task(self):
        with pytest.raises(ValidationError, match="capture: response"):
            TaskDefinition(run="echo hi", capture="response")

    def test_capture_rejects_unknown_value(self):
        with pytest.raises(ValidationError):
            TaskDefinition(user_prompt="hi", capture="bogus")

    def test_extra_fields_allowed(self):
        t = TaskDefinition(future_field="value")
        assert t.model_extra["future_field"] == "value"


class TestMultiModel:
    """Test the multi-model ``models:`` grammar on a task."""

    def test_defaults_single_model(self):
        """Absent ``models`` leaves a task single-model with default policy."""
        t = TaskDefinition(user_prompt="hi")
        assert t.models == []
        assert t.completion == "all"
        assert t.model_concurrency == 0
        entries = t.effective_model_entries()
        assert len(entries) == 1
        assert entries[0].model == ""
        assert entries[0].model_settings == {}

    def test_singular_model_wrapped_in_entry(self):
        """The singular ``model``/``model_settings`` pair maps to one entry."""
        t = TaskDefinition(user_prompt="hi", model="fast", model_settings={"temperature": 0.5})
        entries = t.effective_model_entries()
        assert len(entries) == 1
        assert entries[0].model == "fast"
        assert entries[0].model_settings == {"temperature": 0.5}

    def test_string_list_coerced_to_entries(self):
        """A bare list of names coerces each into a ModelEntry."""
        t = TaskDefinition(user_prompt="hi", models=["gpt_default", "claude_native"])
        assert all(isinstance(e, ModelEntry) for e in t.models)
        assert [e.model for e in t.effective_model_entries()] == ["gpt_default", "claude_native"]
        assert all(e.model_settings == {} for e in t.effective_model_entries())

    def test_rich_entries_preserve_settings(self):
        """Per-entry ``model_settings`` maps are preserved."""
        t = TaskDefinition(
            user_prompt="hi",
            models=[
                {"model": "gpt_default", "model_settings": {"temperature": 0.2}},
                {"model": "claude_native", "model_settings": {"reasoning": {"effort": "high"}}},
            ],
        )
        entries = t.effective_model_entries()
        assert entries[0].model == "gpt_default"
        assert entries[0].model_settings == {"temperature": 0.2}
        assert entries[1].model_settings == {"reasoning": {"effort": "high"}}

    def test_mixed_string_and_map_entries(self):
        """A list may mix bare names and override maps."""
        t = TaskDefinition(
            user_prompt="hi",
            models=["gpt_default", {"model": "claude_native", "model_settings": {"temperature": 0.1}}],
        )
        entries = t.effective_model_entries()
        assert entries[0].model == "gpt_default"
        assert entries[0].model_settings == {}
        assert entries[1].model == "claude_native"
        assert entries[1].model_settings == {"temperature": 0.1}

    def test_model_and_models_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            TaskDefinition(user_prompt="hi", model="fast", models=["a"])

    def test_multi_model_with_repeat_prompt_allowed(self):
        """Multi-model + repeat_prompt is a supported cross product."""
        t = TaskDefinition(user_prompt="{{ result }}", repeat_prompt=True, models=["a", "b"])
        assert len(t.effective_model_entries()) == 2

    def test_single_model_with_repeat_prompt_allowed(self):
        """One-element ``models`` is fine with repeat_prompt (no fan-out)."""
        t = TaskDefinition(user_prompt="{{ result }}", repeat_prompt=True, models=["only"])
        assert len(t.effective_model_entries()) == 1

    def test_completion_policy_validated(self):
        with pytest.raises(ValidationError):
            TaskDefinition(user_prompt="hi", models=["a", "b"], completion="quorum")
        t = TaskDefinition(user_prompt="hi", models=["a", "b"], completion="any")
        assert t.completion == "any"

    def test_negative_model_concurrency_rejected(self):
        with pytest.raises(ValidationError, match="model_concurrency"):
            TaskDefinition(user_prompt="hi", models=["a", "b"], model_concurrency=-1)

    def test_invalid_models_type_rejected(self):
        with pytest.raises(ValidationError, match="must be a list"):
            TaskDefinition(user_prompt="hi", models="gpt_default")

    def test_invalid_models_entry_rejected(self):
        with pytest.raises(ValidationError, match="invalid 'models' entry"):
            TaskDefinition(user_prompt="hi", models=[123])


class TestTypedOutputs:
    """Test the typed named outputs grammar (id / outputs / over)."""

    def test_conditional_if_field_alias(self):
        """`if:` in YAML maps to the aliased if_ field; defaults to empty."""
        t = TaskDefinition(**{"if": "globals.enabled", "user_prompt": "hi"})
        assert t.if_ == "globals.enabled"
        assert TaskDefinition(user_prompt="hi").if_ == ""

    def test_defaults(self):
        t = TaskDefinition(user_prompt="hi")
        assert t.id == ""
        assert t.outputs == {}
        assert t.over == ""

    def test_id_and_outputs_accepted(self):
        t = TaskDefinition(
            id="list_functions",
            user_prompt="list them",
            outputs={
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
            },
        )
        assert t.id == "list_functions"
        assert t.outputs["properties"]["functions"]["type"] == "array"

    def test_invalid_outputs_schema_rejected_at_load(self):
        with pytest.raises(ValidationError, match="invalid 'outputs' schema"):
            TaskDefinition(id="x", user_prompt="hi", outputs={"type": "not-a-json-schema-type"})

    def test_over_requires_repeat_prompt(self):
        with pytest.raises(ValidationError, match="'over' only applies to repeat_prompt"):
            TaskDefinition(user_prompt="{{ result }}", over="outputs.x.items")

    def test_over_with_repeat_prompt_ok(self):
        t = TaskDefinition(user_prompt="{{ result }}", repeat_prompt=True, over="outputs.x.items")
        assert t.over == "outputs.x.items"

    def test_id_allowed_on_multi_model_for_fanin(self):
        """`id` on a multi-model task is allowed (enables fan-in)."""
        t = TaskDefinition(id="cmp", user_prompt="hi", models=["a", "b"])
        assert t.id == "cmp"

    def test_over_allowed_on_multi_model_repeat(self):
        """`over` (iterable selector) is allowed on a multi-model repeat task."""
        t = TaskDefinition(
            user_prompt="{{ result }}", repeat_prompt=True, models=["a", "b"], over="outputs.x.items"
        )
        assert t.over == "outputs.x.items"

    def test_outputs_schema_allowed_on_multi_model(self):
        # Typed outputs are now supported on multi-model tasks: the schema is
        # applied per branch (see runner fan-in), so construction must succeed.
        schema = {"type": "object", "properties": {"f": {"type": "string"}}, "required": ["f"]}
        t = TaskDefinition(user_prompt="hi", models=["a", "b"], outputs=schema)
        assert t.outputs == schema
        assert [e.model for e in t.models] == ["a", "b"]


class TestTaskflowDocument:
    """Test complete taskflow document parsing."""

    def test_minimal(self):
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "taskflow"},
            "taskflow": [
                {"task": {"agents": ["p.a"], "user_prompt": "Hello"}},
            ],
        }
        doc = TaskflowDocument(**data)
        assert doc.header.filetype == "taskflow"
        assert len(doc.taskflow) == 1
        assert doc.taskflow[0].task.user_prompt == "Hello"

    def test_with_globals_and_model_config(self):
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "taskflow"},
            "globals": {"fruit": "bananas"},
            "model_config": "examples.model_configs.model_config",
            "taskflow": [],
        }
        doc = TaskflowDocument(**data)
        assert doc.globals == {"fruit": "bananas"}
        assert doc.model_config_ref == "examples.model_configs.model_config"

    def test_null_taskflow(self):
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "taskflow"},
            "taskflow": None,
        }
        doc = TaskflowDocument(**data)
        assert doc.taskflow == []

    def test_integer_version(self):
        data = {
            "seclab-taskflow-agent": {"version": 1, "filetype": "taskflow"},
            "taskflow": [],
        }
        doc = TaskflowDocument(**data)
        assert doc.header.version == "1.0"


class TestPersonalityDocument:
    """Test personality document parsing."""

    def test_full_personality(self):
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "personality"},
            "personality": "You are a helpful assistant.\n",
            "task": "Answer any question.\n",
            "toolboxes": ["seclab_taskflow_agent.toolboxes.memcache"],
        }
        doc = PersonalityDocument(**data)
        assert doc.personality == "You are a helpful assistant.\n"
        assert len(doc.toolboxes) == 1

    def test_minimal_personality(self):
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "personality"},
        }
        doc = PersonalityDocument(**data)
        assert doc.personality == ""
        assert doc.toolboxes == []


class TestToolboxDocument:
    """Test toolbox document parsing."""

    def test_stdio_toolbox(self):
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "toolbox"},
            "server_params": {
                "kind": "stdio",
                "command": "python",
                "args": ["-m", "module.server"],
                "env": {"KEY": "value"},
            },
            "confirm": ["dangerous_tool"],
        }
        doc = ToolboxDocument(**data)
        assert doc.server_params.kind == "stdio"
        assert doc.server_params.command == "python"
        assert doc.confirm == ["dangerous_tool"]

    def test_streamable_toolbox(self):
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "toolbox"},
            "server_params": {
                "kind": "streamable",
                "url": "http://localhost:9999/mcp",
                "command": "python",
                "args": ["-m", "module.server"],
            },
            "server_prompt": "Use this server for queries.",
        }
        doc = ToolboxDocument(**data)
        assert doc.server_params.kind == "streamable"
        assert doc.server_params.url == "http://localhost:9999/mcp"
        assert doc.server_prompt == "Use this server for queries."


class TestModelConfigDocument:
    """Test model config document parsing."""

    def test_full_config(self):
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "model_config"},
            "models": {"gpt_default": "gpt-4.1", "gpt_latest": "gpt-5"},
            "model_settings": {
                "gpt_default": {"temperature": 0.7},
            },
        }
        doc = ModelConfigDocument(**data)
        assert doc.models["gpt_default"] == "gpt-4.1"
        assert doc.model_settings["gpt_default"]["temperature"] == 0.7
        assert doc.api_type == "chat_completions"  # default

    def test_api_type_responses(self):
        """Test that api_type can be set to 'responses'."""
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "model_config"},
            "api_type": "responses",
            "models": {"o3": "o3"},
        }
        doc = ModelConfigDocument(**data)
        assert doc.api_type == "responses"

    def test_api_type_invalid(self):
        """Test that invalid api_type values are rejected."""
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "model_config"},
            "api_type": "invalid",
            "models": {},
        }
        with pytest.raises(ValidationError):
            ModelConfigDocument(**data)

    def test_backend_default_none(self):
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "model_config"},
            "models": {},
        }
        assert ModelConfigDocument(**data).backend is None

    def test_backend_explicit(self):
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "model_config"},
            "backend": "copilot_sdk",
            "models": {},
        }
        assert ModelConfigDocument(**data).backend == "copilot_sdk"

    def test_backend_invalid(self):
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "model_config"},
            "backend": "not_a_backend",
            "models": {},
        }
        with pytest.raises(ValidationError):
            ModelConfigDocument(**data)


class TestPromptDocument:
    """Test prompt document parsing."""

    def test_prompt(self):
        data = {
            "seclab-taskflow-agent": {"version": "1.0", "filetype": "prompt"},
            "prompt": "Tell me about bananas.\n",
        }
        doc = PromptDocument(**data)
        assert doc.prompt == "Tell me about bananas.\n"


class TestServerParams:
    """Test server params validation."""

    def test_extra_fields_allowed(self):
        sp = ServerParams(kind="stdio", custom_field="hello")
        assert sp.model_extra["custom_field"] == "hello"

    def test_minimal(self):
        sp = ServerParams(kind="sse", url="http://localhost:8080")
        assert sp.kind == "sse"
        assert sp.command is None


class TestRealYAMLFiles:
    """Test parsing actual project YAML files through Pydantic models."""

    def test_parse_example_taskflow(self):
        import yaml

        with open("examples/taskflows/example.yaml") as f:
            data = yaml.safe_load(f)
        doc = TaskflowDocument(**data)
        assert len(doc.taskflow) == 4
        assert doc.model_config_ref == "examples.model_configs.model_config"

    def test_parse_echo_taskflow(self):
        import yaml

        with open("examples/taskflows/echo.yaml") as f:
            data = yaml.safe_load(f)
        doc = TaskflowDocument(**data)
        assert len(doc.taskflow) == 2
        assert doc.taskflow[0].task.must_complete is True
        assert doc.taskflow[0].task.max_steps == 5

    def test_parse_example_globals(self):
        import yaml

        with open("examples/taskflows/example_globals.yaml") as f:
            data = yaml.safe_load(f)
        doc = TaskflowDocument(**data)
        assert "fruit" in doc.globals

    def test_parse_personality(self):
        import yaml

        with open("src/seclab_taskflow_agent/personalities/assistant.yaml") as f:
            data = yaml.safe_load(f)
        doc = PersonalityDocument(**data)
        assert doc.personality != ""

    def test_parse_toolbox_memcache(self):
        import yaml

        with open("src/seclab_taskflow_agent/toolboxes/memcache.yaml") as f:
            data = yaml.safe_load(f)
        doc = ToolboxDocument(**data)
        assert doc.server_params.kind == "stdio"
        assert "memcache_clear_cache" in doc.confirm

    def test_parse_toolbox_codeql(self):
        import yaml

        with open("src/seclab_taskflow_agent/toolboxes/codeql.yaml") as f:
            data = yaml.safe_load(f)
        doc = ToolboxDocument(**data)
        assert doc.server_params.kind == "streamable"
        assert doc.server_prompt != ""

    def test_parse_model_config(self):
        import yaml

        with open("examples/model_configs/model_config.yaml") as f:
            data = yaml.safe_load(f)
        doc = ModelConfigDocument(**data)
        assert "gpt_default" in doc.models

    def test_parse_prompt(self):
        import yaml

        with open("examples/prompts/example_prompt.yaml") as f:
            data = yaml.safe_load(f)
        doc = PromptDocument(**data)
        assert "bananas" in doc.prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
