# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for buffered output rendering and per-model stream labelling."""

import asyncio

from seclab_taskflow_agent import render_utils
from seclab_taskflow_agent.render_utils import flush_async_output, render_model_output


def _capture(monkeypatch) -> list[str]:
    """Redirect print() inside render_model_output into a captured list."""
    captured: list[str] = []

    def _fake_print(data, *args, **kwargs):
        captured.append(data)

    monkeypatch.setattr("builtins.print", _fake_print)
    return captured


class TestFlushAsyncOutput:
    """flush_async_output buffers per task and flushes with an optional label."""

    def test_buffered_output_flushes_with_default_heading(self, monkeypatch):
        captured = _capture(monkeypatch)

        async def scenario():
            await render_model_output("hello ", async_task=True, task_id="abc123")
            await render_model_output("world", async_task=True, task_id="abc123")
            await flush_async_output("abc123")

        asyncio.run(scenario())
        joined = "".join(captured)
        # Default heading names the task id.
        assert "Output for async task: abc123" in joined
        assert "hello world" in joined

    def test_label_overrides_heading(self, monkeypatch):
        captured = _capture(monkeypatch)

        async def scenario():
            await render_model_output("audit result", async_task=True, task_id="tid-1")
            await flush_async_output("tid-1", label="model: gpt_default")

        asyncio.run(scenario())
        joined = "".join(captured)
        assert "Output for model: gpt_default" in joined
        # The task id is not used when a label is supplied.
        assert "async task: tid-1" not in joined
        assert "audit result" in joined

    def test_flush_unknown_task_is_noop(self, monkeypatch):
        captured = _capture(monkeypatch)
        asyncio.run(flush_async_output("does-not-exist", label="model: x"))
        assert captured == []

    def test_buffers_are_isolated_per_task(self, monkeypatch):
        captured = _capture(monkeypatch)

        async def scenario():
            await render_model_output("A-out", async_task=True, task_id="A")
            await render_model_output("B-out", async_task=True, task_id="B")
            await flush_async_output("A", label="model: alpha")
            await flush_async_output("B", label="model: beta")

        asyncio.run(scenario())
        joined = "".join(captured)
        assert "model: alpha" in joined
        assert "model: beta" in joined
        assert "A-out" in joined
        assert "B-out" in joined
        # Both buffers fully drained.
        assert render_utils.async_output == {}
