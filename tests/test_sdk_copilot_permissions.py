# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``sdk.copilot_sdk.permissions``."""

from __future__ import annotations

import pytest

pytest.importorskip("copilot")

from copilot.rpc import PermissionDecisionApproveOnce, PermissionDecisionReject
from copilot.session_events import (
    PermissionRequestCustomTool,
    PermissionRequestHook,
    PermissionRequestMcp,
    PermissionRequestMemory,
    PermissionRequestRead,
    PermissionRequestShell,
    PermissionRequestUrl,
    PermissionRequestWrite,
)

from seclab_taskflow_agent.sdk.copilot_sdk.permissions import build_permission_handler


def _mcp(tool_name: str = "safe", **kw) -> PermissionRequestMcp:
    return PermissionRequestMcp(
        read_only=False,
        server_name="toolbox",
        tool_name=tool_name,
        tool_title=tool_name,
        **kw,
    )


def _shell(command: str = "ls") -> PermissionRequestShell:
    return PermissionRequestShell(
        can_offer_session_approval=False,
        commands=[],
        full_command_text=command,
        has_write_file_redirection=False,
        intention="run",
        possible_paths=[],
        possible_urls=[],
    )


def _write(path: str = "/tmp/x") -> PermissionRequestWrite:
    return PermissionRequestWrite(can_offer_session_approval=False, diff="+x", file_name=path, intention="edit")


@pytest.mark.parametrize(
    "request_",
    [
        _mcp(),
        PermissionRequestRead(intention="read", path="/etc/hostname"),
        PermissionRequestUrl(intention="fetch", url="https://example.com"),
        _shell(),
        _write(),
        PermissionRequestMemory(fact="remember me"),
        PermissionRequestHook(tool_name="h"),
    ],
    ids=lambda r: r.kind,
)
def test_raw_mcp_and_builtin_permissions_are_denied(request_):
    result = build_permission_handler(["safe"])(request_, {"session_id": "s"})
    assert isinstance(result, PermissionDecisionReject)


def test_only_registered_policy_enforcing_tools_are_approved():
    handler = build_permission_handler(["safe"])
    result = handler(
        PermissionRequestCustomTool(tool_description="d", tool_name="safe"),
        {"session_id": "s", "managed_settings_enabled": False},
    )
    assert isinstance(result, PermissionDecisionApproveOnce)
    assert result.to_dict() == {"kind": "approve-once"}


def test_unknown_kind_is_denied():
    class _Future:
        kind = "something-new"

    result = build_permission_handler([])(_Future(), {"session_id": "s"})
    assert isinstance(result, PermissionDecisionReject)


def test_unregistered_custom_tool_is_rejected():
    result = build_permission_handler(["safe"])(
        PermissionRequestCustomTool(tool_description="d", tool_name="dangerous"),
        {"session_id": "s"},
    )
    assert isinstance(result, PermissionDecisionReject)


def test_managed_approval_required_is_rejected():
    result = build_permission_handler(["safe"])(
        PermissionRequestCustomTool(
            tool_description="d",
            tool_name="safe",
            managed_approval_required=True,
        ),
        {"session_id": "s"},
    )
    assert isinstance(result, PermissionDecisionReject)
