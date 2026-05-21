# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Permission handling for the Copilot SDK backend.

The Copilot SDK requires every session to provide an
``on_permission_request`` callback. The taskflow YAML exposes a single
policy knob, ``blocked_tools``: any tool whose name or raw
command/path/url appears in that list is denied (exact string match);
everything else is approved. The ``headless`` flag is accepted for
signature parity with the openai-agents path but is unused — the
runner never has a TTY, so there is no interactive fallback to prompt
for.
"""

from __future__ import annotations

__all__ = ["build_permission_handler"]

from collections.abc import Callable
from typing import Any


def _request_field(request: Any, *names: str) -> str | None:
    for name in names:
        value = getattr(request, name, None)
        if value:
            return str(value)
    return None


def build_permission_handler(
    blocked_tools: list[str],
    *,
    headless: bool,
) -> Callable[[Any, dict[str, str]], Any]:
    # Imported lazily so test code that does not exercise the handler
    # can import this module without the SDK installed.
    from copilot.session import PermissionRequestResult

    del headless  # see module docstring
    blocked = set(blocked_tools)

    def _handler(request: Any, _invocation: dict[str, str]) -> Any:
        tool_id = _request_field(request, "tool_name", "full_command_text", "path", "url")
        if tool_id and tool_id in blocked:
            return PermissionRequestResult(
                kind="denied-by-rules",
                message=f"Tool {tool_id!r} is blocked by taskflow configuration",
            )
        return PermissionRequestResult(kind="approved")

    return _handler
