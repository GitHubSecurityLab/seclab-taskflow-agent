# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``sdk.copilot_sdk.permissions``."""

from __future__ import annotations

import pytest

pytest.importorskip("copilot")

from dataclasses import dataclass

from seclab_taskflow_agent.sdk.copilot_sdk.permissions import build_permission_handler


@dataclass
class _Req:
    tool_name: str | None = None
    full_command_text: str | None = None
    path: str | None = None
    url: str | None = None


def test_blocked_tool_is_denied_by_rules():
    handler = build_permission_handler(["dangerous"], headless=True)
    result = handler(_Req(tool_name="dangerous"), {})
    assert result.kind == "denied-by-rules"
    assert "dangerous" in (result.message or "")


def test_headless_approves_unblocked_tool():
    handler = build_permission_handler(["other"], headless=True)
    result = handler(_Req(tool_name="safe"), {})
    assert result.kind == "approved"


def test_non_headless_still_approves_unblocked_tool():
    # The Copilot SDK requires a permission verdict; the runner has no
    # interactive TTY, so any non-blocked tool is approved regardless of
    # the YAML ``headless`` flag (the openai-agents path has the same
    # effective behaviour).
    handler = build_permission_handler([], headless=False)
    result = handler(_Req(tool_name="safe"), {})
    assert result.kind == "approved"


def test_block_matches_command_text_when_tool_name_missing():
    handler = build_permission_handler(["rm -rf /"], headless=True)
    result = handler(_Req(full_command_text="rm -rf /"), {})
    assert result.kind == "denied-by-rules"


def test_handler_signature_matches_sdk_callback() -> None:
    # Sanity check: the SDK calls handler(request, invocation); make sure
    # we accept exactly two positional arguments.
    handler = build_permission_handler([], headless=True)
    handler(_Req(tool_name="ok"), {"invocation": "x"})  # type: ignore[arg-type]
