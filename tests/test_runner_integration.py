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
from unittest.mock import MagicMock

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


def _run(monkeypatch, tmp_path, doc: TaskflowDocument) -> list[dict]:
    """Run a taskflow with deploy patched; return the recorded deploy calls."""
    from seclab_taskflow_agent import runner
    from seclab_taskflow_agent.runner import run_main

    calls: list[dict] = []

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
        if record_tool_result is not None:
            await record_tool_result(ToolResult(text=json.dumps({"model": model, "prompt": prompt})))
        return True

    monkeypatch.setattr(runner, "deploy_task_agents", fake_deploy)
    monkeypatch.setattr(runner, "start_watchdog", lambda: None)
    monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)

    at = MagicMock()
    at.get_taskflow.return_value = doc
    at.get_personality.return_value = _personality()

    asyncio.run(run_main(at, None, "pkg.flow", {}, None))
    return calls


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
