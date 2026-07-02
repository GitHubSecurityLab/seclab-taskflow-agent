# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the run manifest (structured audit view + on-disk artifact)."""

import json

from seclab_taskflow_agent.session import TaskflowSession, artifacts_dir


class TestManifest:
    def test_manifest_structure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="pkg.flow", total_tasks=2)
        s.record_task(index=0, name="audit", success=True, models=["m1", "m2"], duration_s=1.2345,
                      result_snapshot={"results": [], "outputs": {"audit": [{"model": "m1", "item": 0, "result": 1}]}})
        s.record_task(index=1, name="skipme", success=True, skipped=True)

        m = s.manifest()
        assert m["session_id"] == s.session_id
        assert m["taskflow"] == "pkg.flow"
        assert m["status"] == "in_progress"
        assert m["total_tasks"] == 2
        assert [t["name"] for t in m["tasks"]] == ["audit", "skipme"]
        assert m["tasks"][0]["status"] == "ok"
        assert m["tasks"][0]["models"] == ["m1", "m2"]
        assert m["tasks"][0]["duration_s"] == 1.234  # rounded
        assert m["tasks"][1]["status"] == "skipped"
        assert m["outputs"]["audit"][0]["model"] == "m1"

    def test_status_transitions(self, tmp_path, monkeypatch):
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="pkg.flow")
        assert s.status == "in_progress"
        s.mark_failed("boom")
        assert s.status == "failed"
        assert s.manifest()["error"] == "boom"

        s2 = TaskflowSession(taskflow_path="pkg.flow")
        s2.mark_finished()
        assert s2.status == "finished"
        assert s2.finished_at

    def test_mark_finished_writes_manifest_artifact(self, tmp_path, monkeypatch):
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="pkg.flow", total_tasks=1)
        s.record_task(index=0, name="t0", success=True, models=["m1"], duration_s=0.5)
        s.mark_finished()

        path = artifacts_dir(s.session_id) / "manifest.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["status"] == "finished"
        assert data["tasks"][0]["models"] == ["m1"]

    def test_mark_failed_writes_manifest_artifact(self, tmp_path, monkeypatch):
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="pkg.flow")
        s.mark_failed("nope")
        data = json.loads((artifacts_dir(s.session_id) / "manifest.json").read_text())
        assert data["status"] == "failed"
        assert data["error"] == "nope"


class TestManifestCLI:
    def test_print_manifest_ok(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        from seclab_taskflow_agent import cli

        s = TaskflowSession(taskflow_path="pkg.flow", total_tasks=1)
        s.record_task(index=0, name="t0", success=True, models=["m1"])
        s.save()

        rc = cli._print_manifest(s.session_id)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["session_id"] == s.session_id
        assert out["tasks"][0]["name"] == "t0"

    def test_print_manifest_missing_returns_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        from seclab_taskflow_agent import cli

        assert cli._print_manifest("does-not-exist") == 1


class TestManifestWriteIsolation:
    def test_manifest_write_failure_does_not_break_run(self, tmp_path, monkeypatch):
        """A failing manifest write is swallowed; the checkpoint still saves."""
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="pkg.flow", total_tasks=1)
        s.record_task(index=0, name="t0", success=True, models=["m1"])

        # Force the manifest serialization to fail.
        def _boom(*_a, **_k):
            raise RuntimeError("boom")

        monkeypatch.setattr("seclab_taskflow_agent.session.json.dumps", _boom)
        # Must not raise, and must still mark the session finished + checkpoint it.
        s.mark_finished()
        assert s.finished is True
        loaded = TaskflowSession.load(s.session_id)
        assert loaded.finished is True
