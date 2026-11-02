# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Allow only registered, policy-enforcing taskflow tool handlers.

An SDK approval permits dispatch to the wrapper, not execution without its
confirmation checks. Raw MCP requests, built-in tools, unknown tools and
managed-approval requests are denied, including reads and URL fetches.
"""

from __future__ import annotations

__all__ = ["build_permission_handler"]

from collections.abc import Callable, Collection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from copilot import PermissionRequest, PermissionRequestResult
    from copilot.session import PermissionInvocation


def build_permission_handler(
    allowed_tools: Collection[str],
) -> Callable[[PermissionRequest, PermissionInvocation], PermissionRequestResult]:
    from copilot.rpc import PermissionDecisionApproveOnce, PermissionDecisionReject
    from copilot.session_events import PermissionRequestCustomTool

    allowed = frozenset(allowed_tools)

    def _handler(request: PermissionRequest, _invocation: PermissionInvocation) -> PermissionRequestResult:
        if getattr(request, "managed_approval_required", False):
            return PermissionDecisionReject(feedback="Managed approval is required and cannot be granted by a taskflow")
        if isinstance(request, PermissionRequestCustomTool) and request.tool_name in allowed:
            return PermissionDecisionApproveOnce()
        return PermissionDecisionReject(feedback="Only configured taskflow tools may be invoked")

    return _handler
