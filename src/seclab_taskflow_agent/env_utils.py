# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Environment variable utilities for taskflow execution."""

import os
from typing import Any

import jinja2

__all__ = ["TmpEnv", "swap_env"]



def swap_env(s: str, context: dict[str, Any] | None = None) -> str:
    """Render Jinja template expressions in a string.

    Supports expressions such as ``{{ env('VAR') }}``. Template variables
    like ``{{ globals.X }}`` are only available when provided by the caller
    via ``context`` (e.g. ``{'globals': {...}}``).

    Args:
        s: String potentially containing templates.
        context: Optional template context. Variables such as ``globals``
            must be supplied here to be available during rendering.

    Returns:
        String with templates replaced.

    Raises:
        LookupError: If a required environment variable or template
            variable is not found during rendering.
    """
    try:
        from .template_utils import create_jinja_environment
        from .available_tools import AvailableTools

        available_tools = AvailableTools()
        jinja_env = create_jinja_environment(available_tools)
        template = jinja_env.from_string(s)
        # Filter out keys that collide with built-in template globals
        # (e.g. the env() helper) to prevent callers from breaking them.
        reserved_keys = set(jinja_env.globals)
        render_context = {
            key: value for key, value in (context or {}).items()
            if key not in reserved_keys
        }
        return template.render(**render_context)
    except jinja2.UndefinedError as e:
        raise LookupError(str(e))
    except jinja2.TemplateError as e:
        raise LookupError(f"Template rendering failed for: {s!r}: {e}")


class TmpEnv:
    """Context manager that temporarily sets environment variables."""

    def __init__(self, env: dict[str, str],
                 context: dict[str, Any] | None = None) -> None:
        self.env = dict(env)
        self.context = context
        self.restore_env = dict(os.environ)

    def __enter__(self) -> None:
        applied: list[str] = []
        try:
            for k, v in self.env.items():
                os.environ[k] = swap_env(v, self.context)
                applied.append(k)
        except Exception:
            for k in applied:
                if k in self.restore_env:
                    os.environ[k] = self.restore_env[k]
                else:
                    os.environ.pop(k, None)
            raise

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> None:
        for k, v in self.env.items():
            del os.environ[k]
            if k in self.restore_env:
                os.environ[k] = self.restore_env[k]
