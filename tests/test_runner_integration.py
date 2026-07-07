# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Integration tests driving run_main with a patched deploy layer.

These exercise the multi-model / cross-product fan-out and fan-in end to end
(work matrix, per-branch stream labels, and outputs.<id> aggregation) without
any real agents, MCP servers, or model calls.
"""

from __future__ import annotations

import asyncio
import glob
import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from seclab_taskflow_agent.models import (
    PersonalityDocument,
    TaskDefinition,
    TaskflowDocument,
    TaskflowHeader,
    TaskWrapper,
)
from seclab_taskflow_agent.results import ToolResult


def _personality() -> PersonalityDocument:
    return PersonalityDocument(
        **{
            "seclab-taskflow-agent": TaskflowHeader(version="1.0", filetype="personality"),
            "personality": "p",
            "task": "t",
            "toolboxes": [],
        }
    )


def _taskflow(task: TaskDefinition, globals_: dict | None = None) -> TaskflowDocument:
    return TaskflowDocument(
        **{
            "seclab-taskflow-agent": TaskflowHeader(version="1.0", filetype="taskflow"),
            "globals": globals_ or {},
            "taskflow": [TaskWrapper(task=task)],
        }
    )


def _taskflow_tasks(tasks: list[TaskDefinition], globals_: dict | None = None) -> TaskflowDocument:
    return TaskflowDocument(
        **{
            "seclab-taskflow-agent": TaskflowHeader(version="1.0", filetype="taskflow"),
            "globals": globals_ or {},
            "taskflow": [TaskWrapper(task=t) for t in tasks],
        }
    )


def _run(monkeypatch, tmp_path, doc: TaskflowDocument, deploy_impl=None) -> list[dict]:
    """Run a taskflow with deploy patched; return the recorded deploy calls."""
    from seclab_taskflow_agent import runner
    from seclab_taskflow_agent.runner import run_main

    calls: list[dict] = []

    async def default_deploy(available_tools, agents, prompt, *, model=None, record_tool_result=None, **kw):
        if record_tool_result is not None:
            await record_tool_result(ToolResult(text=json.dumps({"model": model, "prompt": prompt})))
        return True

    async def fake_deploy(
        available_tools,
        agents,
        prompt,
        *,
        model=None,
        stream_label=None,
        record_tool_result=None,
        run_hooks=None,
        **kw,
    ):
        calls.append({"prompt": prompt, "model": model, "label": stream_label})
        impl = deploy_impl or default_deploy
        return await impl(
            available_tools, agents, prompt, model=model, record_tool_result=record_tool_result, **kw
        )

    monkeypatch.setattr(runner, "deploy_task_agents", fake_deploy)
    monkeypatch.setattr(runner, "start_watchdog", lambda: None)
    monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)

    at = MagicMock()
    at.get_taskflow.return_value = doc
    at.get_personality.return_value = _personality()

    asyncio.run(run_main(at, None, "pkg.flow", {}, None))
    return calls


def _session_data(tmp_path) -> dict:
    files = glob.glob(str(tmp_path / "sessions" / "*.json"))
    assert files, "no session file written"
    latest = max(files, key=lambda p: Path(p).stat().st_mtime)
    return json.loads(Path(latest).read_text())


def _session_outputs(tmp_path) -> dict:
    files = glob.glob(str(tmp_path / "sessions" / "*.json"))
    assert files, "no session file written"
    # Most recent session.
    latest = max(files, key=lambda p: Path(p).stat().st_mtime)
    data = json.loads(Path(latest).read_text())
    return (data.get("result_snapshot") or {}).get("outputs") or {}


class TestMultiModelFanout:
    def test_plain_multi_model_runs_each_model_once(self, monkeypatch, tmp_path):
        task = TaskDefinition(
            id="cmp",
            agents=["pkg.p"],
            user_prompt="explain X",
            models=["m1", "m2", "m3"],
        )
        calls = _run(monkeypatch, tmp_path, _taskflow(task))
        # One deploy per model, labelled by the model (no item suffix).
        assert len(calls) == 3
        assert {c["model"] for c in calls} == {"m1", "m2", "m3"}
        assert {c["label"] for c in calls} == {"m1", "m2", "m3"}

    def test_plain_multi_model_fans_in_per_model(self, monkeypatch, tmp_path):
        task = TaskDefinition(id="cmp", agents=["pkg.p"], user_prompt="explain X", models=["m1", "m2"])
        _run(monkeypatch, tmp_path, _taskflow(task))
        outputs = _session_outputs(tmp_path)
        assert "cmp" in outputs
        records = outputs["cmp"]
        assert [r["model"] for r in records] == ["m1", "m2"]
        assert all(r["item"] == 0 for r in records)
        # Each result is the decoded fake tool result for that model.
        assert records[0]["result"] == {"model": "m1", "prompt": "explain X"}


class TestCrossProduct:
    def _cross_doc(self) -> TaskflowDocument:
        task = TaskDefinition(
            id="matrix",
            agents=["pkg.p"],
            user_prompt="rate {{ result }}",
            models=["m1", "m2"],
            repeat_prompt=True,
            over="globals.items",
        )
        return _taskflow(task, globals_={"items": ["x", "y"]})

    def test_cross_product_runs_items_times_models(self, monkeypatch, tmp_path):
        calls = _run(monkeypatch, tmp_path, self._cross_doc())
        # 2 items x 2 models = 4 deploys.
        assert len(calls) == 4
        pairs = {(c["model"], c["prompt"]) for c in calls}
        assert pairs == {
            ("m1", "rate x"),
            ("m2", "rate x"),
            ("m1", "rate y"),
            ("m2", "rate y"),
        }

    def test_cross_product_labels_include_item_index(self, monkeypatch, tmp_path):
        calls = _run(monkeypatch, tmp_path, self._cross_doc())
        labels = {c["label"] for c in calls}
        assert labels == {"m1 [item 0]", "m2 [item 0]", "m1 [item 1]", "m2 [item 1]"}

    def test_cross_product_fans_in_all_cells(self, monkeypatch, tmp_path):
        _run(monkeypatch, tmp_path, self._cross_doc())
        outputs = _session_outputs(tmp_path)
        records = outputs["matrix"]
        assert len(records) == 4
        # Records tagged by model and item index.
        tagged = {(r["model"], r["item"]) for r in records}
        assert tagged == {("m1", 0), ("m2", 0), ("m1", 1), ("m2", 1)}


class TestModelConcurrencyBound:
    def test_model_concurrency_limits_active_branches(self, monkeypatch, tmp_path):
        """model_concurrency bounds how many branches run at once."""
        from seclab_taskflow_agent import runner
        from seclab_taskflow_agent.runner import run_main

        active = 0
        peak = 0

        async def tracking_deploy(available_tools, agents, prompt, *, record_tool_result=None, **kw):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            if record_tool_result is not None:
                await record_tool_result(ToolResult(text="ok"))
            return True

        monkeypatch.setattr(runner, "deploy_task_agents", tracking_deploy)
        monkeypatch.setattr(runner, "start_watchdog", lambda: None)
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)

        task = TaskDefinition(
            agents=["pkg.p"], user_prompt="hi", models=["m1", "m2", "m3", "m4"], model_concurrency=1
        )
        at = MagicMock()
        at.get_taskflow.return_value = _taskflow(task)
        at.get_personality.return_value = _personality()

        asyncio.run(run_main(at, None, "pkg.flow", {}, None))
        assert peak == 1


class TestOutputCaptureFailure:
    def test_schema_violation_marks_session_failed_and_raises(self, monkeypatch, tmp_path):
        """A declared-but-violated output schema fails the run cleanly."""
        from seclab_taskflow_agent import runner
        from seclab_taskflow_agent.runner import run_main

        async def fake_deploy(available_tools, agents, prompt, *, record_tool_result=None, **kw):
            # produce a value that violates outputs: {items: list[str]}
            if record_tool_result is not None:
                await record_tool_result(ToolResult(text=json.dumps({"items": [1, 2, 3]})))
            return True

        monkeypatch.setattr(runner, "deploy_task_agents", fake_deploy)
        monkeypatch.setattr(runner, "start_watchdog", lambda: None)
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)

        task = TaskDefinition(
            id="x", agents=["pkg.p"], user_prompt="hi",
            outputs={
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"type": "string"}}},
                "required": ["items"],
            },
        )
        at = MagicMock()
        at.get_taskflow.return_value = _taskflow(task)
        at.get_personality.return_value = _personality()

        raised = False
        try:
            asyncio.run(run_main(at, None, "pkg.flow", {}, None))
        except Exception:
            raised = True
        assert raised, "run_main should propagate the capture failure"

        # Session must be marked failed (not left silently clean).
        files = glob.glob(str(tmp_path / "sessions" / "*.json"))
        assert files
        data = json.loads(Path(max(files, key=lambda p: Path(p).stat().st_mtime)).read_text())
        assert data.get("error")
        assert "output capture failed" in data["error"]


class TestConditionalExecution:
    """GitHub-Actions-style `if:` gates whether a task runs."""

    def test_task_skipped_when_condition_false(self, monkeypatch, tmp_path):
        task = TaskDefinition(agents=["pkg.p"], user_prompt="hi", **{"if": "globals.enabled"})
        calls = _run(monkeypatch, tmp_path, _taskflow(task, globals_={"enabled": False}))
        assert calls == []  # deploy never invoked
        data = _session_data(tmp_path)
        assert len(data["completed_tasks"]) == 1
        assert data["completed_tasks"][0]["skipped"] is True

    def test_task_runs_when_condition_true(self, monkeypatch, tmp_path):
        task = TaskDefinition(agents=["pkg.p"], user_prompt="hi", **{"if": "globals.enabled"})
        calls = _run(monkeypatch, tmp_path, _taskflow(task, globals_={"enabled": True}))
        assert len(calls) == 1
        assert _session_data(tmp_path)["completed_tasks"][0]["skipped"] is False

    def test_condition_on_prior_task_output_runs(self, monkeypatch, tmp_path):
        # task A produces findings; task B runs only if there are any.
        a = TaskDefinition(id="audit", agents=["pkg.p"], user_prompt="audit")
        b = TaskDefinition(agents=["pkg.p"], user_prompt="remediate", **{"if": "outputs.audit.findings"})
        doc = _taskflow_tasks([a, b])

        async def deploy(available_tools, agents, prompt, *, model=None, record_tool_result=None, **kw):
            payload = {"findings": ["bug"]} if prompt == "audit" else {"ok": True}
            if record_tool_result is not None:
                await record_tool_result(ToolResult(text=json.dumps(payload)))
            return True

        calls = _run(monkeypatch, tmp_path, doc, deploy_impl=deploy)
        # Both tasks ran (audit found something).
        assert {c["prompt"] for c in calls} == {"audit", "remediate"}

    def test_condition_on_prior_task_output_skips(self, monkeypatch, tmp_path):
        a = TaskDefinition(id="audit", agents=["pkg.p"], user_prompt="audit")
        b = TaskDefinition(agents=["pkg.p"], user_prompt="remediate", **{"if": "outputs.audit.findings"})
        doc = _taskflow_tasks([a, b])

        async def deploy(available_tools, agents, prompt, *, model=None, record_tool_result=None, **kw):
            payload = {"findings": []} if prompt == "audit" else {"ok": True}
            if record_tool_result is not None:
                await record_tool_result(ToolResult(text=json.dumps(payload)))
            return True

        calls = _run(monkeypatch, tmp_path, doc, deploy_impl=deploy)
        # Only the audit task ran; remediation was gated out.
        assert [c["prompt"] for c in calls] == ["audit"]
        data = _session_data(tmp_path)
        assert data["completed_tasks"][1]["skipped"] is True


class TestRunManifest:
    def test_run_writes_manifest_with_models_and_status(self, monkeypatch, tmp_path):
        from seclab_taskflow_agent.session import artifacts_dir

        task = TaskDefinition(id="cmp", agents=["pkg.p"], user_prompt="hi", models=["m1", "m2"])
        _run(monkeypatch, tmp_path, _taskflow(task))

        data = _session_data(tmp_path)
        sid = data["session_id"]
        manifest_path = artifacts_dir(sid) / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "finished"
        assert manifest["tasks"][0]["models"] == ["m1", "m2"]
        assert manifest["tasks"][0]["status"] == "ok"
        # fan-in outputs surfaced in the manifest
        assert "cmp" in manifest["outputs"]


class TestConditionalUndefined:
    """`if` on missing context is falsy (skip), not a run-aborting error."""

    def test_if_on_missing_nested_output_skips(self, monkeypatch, tmp_path):
        # No prior task produces `outputs.audit`, so `outputs.audit.findings`
        # references a missing nested attribute -> must skip, not abort.
        task = TaskDefinition(
            agents=["pkg.p"], user_prompt="remediate",
            **{"if": "outputs.audit.findings | length > 0"},
        )
        calls = _run(monkeypatch, tmp_path, _taskflow(task))
        assert calls == []
        data = _session_data(tmp_path)
        assert data["finished"] is True
        assert data["completed_tasks"][0]["skipped"] is True

    def test_if_on_missing_top_level_skips(self, monkeypatch, tmp_path):
        task = TaskDefinition(agents=["pkg.p"], user_prompt="x", **{"if": "outputs.audit"})
        calls = _run(monkeypatch, tmp_path, _taskflow(task))
        assert calls == []
        assert _session_data(tmp_path)["completed_tasks"][0]["skipped"] is True

    def test_if_on_missing_dict_method_key_skips(self, monkeypatch, tmp_path):
        # A missing key named after a dict method (items/keys/values/get) must
        # read as undefined (falsy) -> skip, not resolve to the bound method
        # (which would be truthy and run the task).
        task = TaskDefinition(agents=["pkg.p"], user_prompt="x", **{"if": "globals.cfg.items"})
        calls = _run(monkeypatch, tmp_path, _taskflow(task, globals_={"cfg": {}}))
        assert calls == []
        assert _session_data(tmp_path)["completed_tasks"][0]["skipped"] is True

    def test_upstream_skip_then_downstream_if_still_completes(self, monkeypatch, tmp_path):
        # Upstream `audit` is gated out; downstream `if` reaches into its
        # (now missing) output. The whole run must still finish cleanly.
        a = TaskDefinition(id="audit", agents=["pkg.p"], user_prompt="audit",
                           **{"if": "globals.deep"})
        b = TaskDefinition(agents=["pkg.p"], user_prompt="remediate",
                           **{"if": "outputs.audit.findings"})
        doc = _taskflow_tasks([a, b], globals_={"deep": False})
        calls = _run(monkeypatch, tmp_path, doc)
        assert calls == []  # both skipped
        data = _session_data(tmp_path)
        assert data["finished"] is True
        assert all(t["skipped"] for t in data["completed_tasks"])


class TestTypedMultiModelOutputs:
    """Typed `outputs` schema applied per branch on multi-model tasks."""

    _SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
    }

    @staticmethod
    def _valid_deploy():
        async def _impl(available_tools, agents, prompt, *, model=None, record_tool_result=None, **kw):
            if record_tool_result is not None:
                await record_tool_result(ToolResult(text=json.dumps({"score": 7})))
            return True

        return _impl

    @staticmethod
    def _mixed_deploy():
        # m1 satisfies the schema; every other model violates it (missing `score`).
        async def _impl(available_tools, agents, prompt, *, model=None, record_tool_result=None, **kw):
            if record_tool_result is not None:
                payload = {"score": 1} if model == "m1" else {"wrong": "x"}
                await record_tool_result(ToolResult(text=json.dumps(payload)))
            return True

        return _impl

    def test_typed_fanin_validates_each_branch(self, monkeypatch, tmp_path):
        task = TaskDefinition(
            id="cmp", agents=["pkg.p"], user_prompt="score X",
            models=["m1", "m2"], outputs=self._SCHEMA,
        )
        _run(monkeypatch, tmp_path, _taskflow(task), deploy_impl=self._valid_deploy())
        records = _session_outputs(tmp_path)["cmp"]
        assert [r["model"] for r in records] == ["m1", "m2"]
        # Each branch result is the validated schema value, not the raw envelope.
        assert all(r["result"] == {"score": 7} for r in records)
        assert _session_data(tmp_path)["completed_tasks"][0]["result"] is True

    def test_schema_violation_fails_branch_completion_all(self, monkeypatch, tmp_path):
        task = TaskDefinition(
            id="cmp", agents=["pkg.p"], user_prompt="score X",
            models=["m1", "m2"], outputs=self._SCHEMA, completion="all",
        )
        _run(monkeypatch, tmp_path, _taskflow(task), deploy_impl=self._mixed_deploy())
        by_model = {r["model"]: r["result"] for r in _session_outputs(tmp_path)["cmp"]}
        assert by_model["m1"] == {"score": 1}
        assert by_model["m2"] is None  # violating branch stored as None
        # completion=all: the invalid branch fails the whole task.
        assert _session_data(tmp_path)["completed_tasks"][0]["result"] is False

    def test_schema_violation_tolerated_completion_any(self, monkeypatch, tmp_path):
        task = TaskDefinition(
            id="cmp", agents=["pkg.p"], user_prompt="score X",
            models=["m1", "m2"], outputs=self._SCHEMA, completion="any",
        )
        _run(monkeypatch, tmp_path, _taskflow(task), deploy_impl=self._mixed_deploy())
        # completion=any: one schema-valid branch is enough for task success.
        assert _session_data(tmp_path)["completed_tasks"][0]["result"] is True


class TestMustCompleteFailure:
    """A failed `must_complete` task aborts non-silently so the CLI exits non-zero."""

    def test_must_complete_failure_raises_and_marks_failed(self, monkeypatch, tmp_path):
        async def failing_deploy(available_tools, agents, prompt, *, model=None, record_tool_result=None, **kw):
            return False  # the task never completes

        task = TaskDefinition(agents=["pkg.p"], user_prompt="hi", must_complete=True)
        with pytest.raises(RuntimeError, match="did not complete"):
            _run(monkeypatch, tmp_path, _taskflow(task), deploy_impl=failing_deploy)

        # The session is still saved and marked failed (so --resume works).
        data = _session_data(tmp_path)
        assert data["finished"] is False
        assert "did not complete" in data["error"]


class TestUnifiedCapture:
    """The unified capture model: repeat_prompt and multi-model both fan in,
    plain tasks stay scalar, and single-model results still feed carry-over."""

    def test_single_model_repeat_fans_in_per_item(self, monkeypatch, tmp_path):
        async def deploy(available_tools, agents, prompt, *, model=None, record_tool_result=None, **kw):
            if record_tool_result is not None:
                await record_tool_result(ToolResult(text=json.dumps({"echo": prompt})))
            return True

        task = TaskDefinition(
            id="items", agents=["pkg.p"], user_prompt="do {{ result }}",
            repeat_prompt=True, over="globals.xs",
        )
        _run(monkeypatch, tmp_path, _taskflow(task, globals_={"xs": ["a", "b", "c"]}), deploy_impl=deploy)
        records = _session_outputs(tmp_path)["items"]
        # A per-item fan-in list (previously only the last item was captured).
        assert [r["item"] for r in records] == [0, 1, 2]
        assert records[0]["result"] == {"echo": "do a"}
        assert records[2]["result"] == {"echo": "do c"}

    def test_plain_single_task_output_is_scalar(self, monkeypatch, tmp_path):
        async def deploy(available_tools, agents, prompt, *, model=None, record_tool_result=None, **kw):
            if record_tool_result is not None:
                await record_tool_result(ToolResult(text=json.dumps({"answer": 42})))
            return True

        task = TaskDefinition(id="v", agents=["pkg.p"], user_prompt="hi")
        _run(monkeypatch, tmp_path, _taskflow(task), deploy_impl=deploy)
        # A task that does not fan out publishes its single value, not a list.
        assert _session_outputs(tmp_path)["v"] == {"answer": 42}

    def test_single_model_result_feeds_implicit_carryover(self, monkeypatch, tmp_path):
        calls: list[str] = []

        async def deploy(available_tools, agents, prompt, *, model=None, record_tool_result=None, **kw):
            calls.append(prompt)
            if record_tool_result is not None:
                payload = ["x", "y"] if prompt.startswith("produce") else {"echo": prompt}
                await record_tool_result(ToolResult(text=json.dumps(payload)))
            return True

        tasks = [
            TaskDefinition(agents=["pkg.p"], user_prompt="produce the list"),
            TaskDefinition(agents=["pkg.p"], user_prompt="handle {{ result }}", repeat_prompt=True),
        ]
        _run(monkeypatch, tmp_path, _taskflow_tasks(tasks), deploy_impl=deploy)
        # Task B repeated once per element of Task A's carried-over result, which
        # only works if Task A's branch result was projected into the store.
        assert [c for c in calls if c.startswith("handle")] == ["handle x", "handle y"]

    def test_async_repeat_fanin_is_ordered_despite_completion_order(self, monkeypatch, tmp_path):
        async def deploy(available_tools, agents, prompt, *, model=None, record_tool_result=None, **kw):
            # Earlier items sleep longer so they finish last; the fan-in must
            # still be in item (submission) order, not completion order.
            n = int(prompt.split()[-1])
            await asyncio.sleep((3 - n) * 0.01)
            if record_tool_result is not None:
                await record_tool_result(ToolResult(text=json.dumps({"n": n})))
            return True

        task = TaskDefinition(
            id="items", agents=["pkg.p"], user_prompt="item {{ result }}",
            repeat_prompt=True, over="globals.xs", async_task=True, async_limit=3,
        )
        _run(monkeypatch, tmp_path, _taskflow(task, globals_={"xs": [0, 1, 2]}), deploy_impl=deploy)
        records = _session_outputs(tmp_path)["items"]
        assert [r["result"]["n"] for r in records] == [0, 1, 2]

    def test_shell_task_never_fans_out(self, monkeypatch, tmp_path):
        # A shell task produces no branches; even with `models` set (which makes
        # multi_model true) it captures its single shell result as a scalar, not
        # an empty fan-in list -- the `not run` guard on fans_out.
        task = TaskDefinition(id="sh", run='echo \'{"x": 1}\'', models=["a", "b"])
        _run(monkeypatch, tmp_path, _taskflow(task))
        assert _session_outputs(tmp_path)["sh"] == {"x": 1}
