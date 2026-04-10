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
        s.record_task(index=0, name="task-0", success=True, tool_results=["r1"])
        assert s.next_task_index == 1
        assert s.last_tool_results == ["r1"]
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


class TestCompletedTask:
    """Tests for CompletedTask model."""

    def test_defaults(self):
        t = CompletedTask(index=0)
        assert t.name == ""
        assert t.result is False
        assert t.tool_results == []

    def test_with_results(self):
        t = CompletedTask(index=2, name="analyze", result=True, tool_results=["r1", "r2"])
        assert t.index == 2
        assert t.tool_results == ["r1", "r2"]


class TestSessionForensics:
    """Tests for failure forensics (memcache_snapshot, tool_log_snapshot)."""

    def test_mark_failed_with_snapshot_roundtrips(self, tmp_path, monkeypatch):
        monkeypatch.setattr("seclab_taskflow_agent.session.session_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="test.flow")
        snap = {"findings": ["xss", "sqli"]}
        log = [{"turn": 1, "tool": "search_code", "result_preview": "found"}]
        s.mark_failed("crash", memcache_snapshot=snap, tool_log_snapshot=log)

        loaded = TaskflowSession.load(s.session_id)
        assert loaded.memcache_snapshot == snap
        assert loaded.tool_log_snapshot == log
        assert loaded.error == "crash"

    def test_mark_failed_without_snapshot_backward_compatible(self):
        s = TaskflowSession(taskflow_path="test.flow")
        s.mark_failed("simple error")
        assert s.memcache_snapshot == {}
        assert s.tool_log_snapshot == []
        assert s.error == "simple error"

    def test_old_session_json_without_new_fields_loads(self, tmp_path, monkeypatch):
        """Session JSON from before forensics fields was added still loads."""
        monkeypatch.setattr("seclab_taskflow_agent.session.session_dir", lambda: tmp_path)
        # Write minimal JSON without new fields
        import json

        old_data = {
            "session_id": "oldformat123",
            "taskflow_path": "old.flow",
            "cli_globals": {},
            "prompt": "",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "",
            "completed_tasks": [],
            "total_tasks": 0,
            "finished": False,
            "error": "old error",
            "last_tool_results": [],
        }
        path = tmp_path / "oldformat123.json"
        path.write_text(json.dumps(old_data))

        loaded = TaskflowSession.load("oldformat123")
        assert loaded.error == "old error"
        assert loaded.memcache_snapshot == {}
        assert loaded.tool_log_snapshot == []
