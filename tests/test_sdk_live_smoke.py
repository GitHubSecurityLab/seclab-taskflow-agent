# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Live integration smoke tests for both backends.

These tests are skipped unless the matching environment variable is
set, so the regular CI run stays hermetic. Opt in with:

    SECLAB_TASKFLOW_LIVE_OPENAI_TOKEN=...   # exercises openai_agents
    SECLAB_TASKFLOW_LIVE_COPILOT_TOKEN=...  # exercises copilot_sdk

Each test issues a single trivial prompt and asserts that the backend
streams *something* back without raising. They are intentionally not
sensitive to wording — they prove the wiring is intact.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from seclab_taskflow_agent.sdk import get_backend
from seclab_taskflow_agent.sdk.base import AgentSpec, TextDelta


def _run_smoke(backend_name: str, *, token_env: str, model: str) -> list[str]:
    backend = get_backend(backend_name)
    spec = AgentSpec(
        name="smoke",
        instructions="You are a terse assistant. Reply with exactly one short sentence.",
        model=model,
        token_env=token_env,
        headless=True,
    )

    async def _go():
        handle = await backend.build(spec)
        try:
            chunks = [
                event.text
                async for event in backend.run_streamed(handle, "Say hi.", max_turns=4)
                if isinstance(event, TextDelta)
            ]
            return chunks
        finally:
            await backend.aclose(handle)

    return asyncio.run(_go())


@pytest.mark.skipif(
    not os.getenv("SECLAB_TASKFLOW_LIVE_OPENAI_TOKEN"),
    reason="set SECLAB_TASKFLOW_LIVE_OPENAI_TOKEN to run",
)
def test_openai_agents_live_smoke():
    chunks = _run_smoke(
        "openai_agents",
        token_env="SECLAB_TASKFLOW_LIVE_OPENAI_TOKEN",  # noqa: S106 - env var name, not a secret
        model=os.getenv("SECLAB_TASKFLOW_LIVE_OPENAI_MODEL", "gpt-4.1-mini"),
    )
    assert "".join(chunks).strip(), "expected at least one streamed text chunk"


@pytest.mark.skipif(
    not os.getenv("SECLAB_TASKFLOW_LIVE_COPILOT_TOKEN"),
    reason="set SECLAB_TASKFLOW_LIVE_COPILOT_TOKEN to run",
)
def test_copilot_sdk_live_smoke():
    pytest.importorskip("copilot")
    chunks = _run_smoke(
        "copilot_sdk",
        token_env="SECLAB_TASKFLOW_LIVE_COPILOT_TOKEN",  # noqa: S106 - env var name, not a secret
        model=os.getenv("SECLAB_TASKFLOW_LIVE_COPILOT_MODEL", "gpt-5-mini"),
    )
    assert "".join(chunks).strip(), "expected at least one streamed text chunk"
