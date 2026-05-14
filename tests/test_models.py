# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for Pydantic grammar models."""

import pytest
from pydantic import ValidationError

from seclab_taskflow_agent.models import (
    ModelConfigDocument,
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

    def test_extra_fields_allowed(self):
        t = TaskDefinition(future_field="value")
        assert t.model_extra["future_field"] == "value"


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
