# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the session checkpoint/resume module."""

import pytest

from seclab_taskflow_agent.session import CompletedTask, TaskflowSession


class TestTaskflowSession:
    """Tests for TaskflowSession persistence."""

    def test_create_session(self):
        """A new session gets a unique ID and starts at task 0."""
        s = TaskflowSession(taskflow_path="examples.taskflows.echo")
        assert len(s.session_id) == 12
        assert s.next_task_index == 0
        assert s.finished is False
        assert s.error == ""

    def test_record_task_advances_index(self):
        """Recording a task increments next_task_index."""
        s = TaskflowSession(taskflow_path="test.flow")
        s.record_task(index=0, name="task-0", success=True, result_snapshot={"results": [], "outputs": {"a": 1}})
        assert s.next_task_index == 1
        assert s.result_snapshot == {"results": [], "outputs": {"a": 1}}
        s.record_task(index=1, name="task-1", success=True)
        assert s.next_task_index == 2

    def test_save_and_load(self, tmp_path, monkeypatch):
        """Session can round-trip through JSON on disk."""
        monkeypatch.setattr("seclab_taskflow_agent.session.session_dir", lambda: tmp_path)
        s = TaskflowSession(
            taskflow_path="examples.taskflows.echo",
            cli_globals={"FOO": "bar"},
            total_tasks=3,
        )
        s.record_task(index=0, name="first", success=True)
        s.save()

        loaded = TaskflowSession.load(s.session_id)
        assert loaded.session_id == s.session_id
        assert loaded.taskflow_path == "examples.taskflows.echo"
        assert loaded.next_task_index == 1
        assert loaded.cli_globals == {"FOO": "bar"}

    def test_load_missing_raises(self, tmp_path, monkeypatch):
        """Loading a non-existent session raises FileNotFoundError."""
        monkeypatch.setattr("seclab_taskflow_agent.session.session_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError):
            TaskflowSession.load("nonexistent")

    def test_mark_finished(self):
        """mark_finished sets the finished flag."""
        s = TaskflowSession(taskflow_path="test.flow")
        assert s.finished is False
        s.mark_finished()
        assert s.finished is True

    def test_mark_failed(self):
        """mark_failed records the error message."""
        s = TaskflowSession(taskflow_path="test.flow")
        s.mark_failed("something broke")
        assert s.error == "something broke"
        assert s.finished is False

    def test_list_sessions(self, tmp_path, monkeypatch):
        """list_sessions returns all saved sessions."""
        monkeypatch.setattr("seclab_taskflow_agent.session.session_dir", lambda: tmp_path)
        s1 = TaskflowSession(taskflow_path="flow1")
        s2 = TaskflowSession(taskflow_path="flow2")
        s1.save()
        s2.save()

        sessions = TaskflowSession.list_sessions()
        ids = {s.session_id for s in sessions}
        assert s1.session_id in ids
        assert s2.session_id in ids

    def test_cli_model_config_persisted(self, tmp_path, monkeypatch):
        """cli_model_config is persisted and restored on load."""
        monkeypatch.setattr("seclab_taskflow_agent.session.session_dir", lambda: tmp_path)
        s = TaskflowSession(
            taskflow_path="examples.taskflows.echo",
            cli_model_config="custom.models.fast",
        )
        s.save()

        loaded = TaskflowSession.load(s.session_id)
        assert loaded.cli_model_config == "custom.models.fast"

    def test_cli_model_config_defaults_empty(self):
        """cli_model_config defaults to empty string."""
        s = TaskflowSession(taskflow_path="test.flow")
        assert s.cli_model_config == ""


class TestCompletedTask:
    """Tests for CompletedTask model."""

    def test_defaults(self):
        t = CompletedTask(index=0)
        assert t.name == ""
        assert t.result is False

    def test_with_result(self):
        t = CompletedTask(index=2, name="analyze", result=True)
        assert t.index == 2
        assert t.result is True
