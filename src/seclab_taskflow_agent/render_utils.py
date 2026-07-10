# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Rendering and buffering of streamed model output.

Output routing (live printing plus per-branch buffering for concurrent
async / multi-model runs) is owned by :class:`OutputRouter` instead of module
-global state. The active router is resolved from a
:class:`~contextvars.ContextVar`, so each run gets its own isolated buffers
while the free-function API (:func:`render_model_output` /
:func:`flush_async_output`) that call sites use stays unchanged.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging

from .path_utils import log_file_name

__all__ = [
    "OutputRouter",
    "flush_async_output",
    "get_output_router",
    "render_model_output",
    "use_output_router",
]

render_logger = logging.getLogger("render")
file_handler = logging.FileHandler(log_file_name("render_stdout.log"))
file_handler.terminator = ""
render_logger.addHandler(file_handler)
render_logger.propagate = False


class OutputRouter:
    """Owns buffered-output state for one run.

    Live output is printed immediately; output tagged with ``async_task`` and a
    ``task_id`` is buffered per branch and flushed as a labelled block when the
    branch completes, so concurrent async / multi-model streams do not
    interleave. One instance per run keeps these buffers isolated (previously
    this was a module-global dict).
    """

    def __init__(self) -> None:
        self._buffers: dict[str, str] = {}
        self._lock = asyncio.Lock()

    @property
    def buffers(self) -> dict[str, str]:
        """The per-branch buffer map (primarily for tests/introspection)."""
        return self._buffers

    async def render(
        self,
        data: str,
        *,
        log: bool = True,
        async_task: bool = False,
        task_id: str | None = None,
    ) -> None:
        """Print *data*, buffering it per ``task_id`` for async branches."""
        async with self._lock:
            if async_task and task_id:
                if task_id in self._buffers:
                    self._buffers[task_id] += data
                    data = ""
                else:
                    self._buffers[task_id] = data
                    data = "** 🤖✏️ Gathering output from async task ... please hold\n"
        if data:
            if log:
                render_logger.info(data)
            print(data, end="", flush=True)

    async def flush(self, task_id: str, label: str | None = None) -> None:
        """Flush the buffered output for *task_id* as a labelled block."""
        async with self._lock:
            if task_id not in self._buffers:
                # No buffered output (agent may have failed before producing any).
                return
            data = self._buffers.pop(task_id)
        heading = label if label else f"async task: {task_id}"
        await self.render(f"** 🤖✏️ Output for {heading}\n\n")
        await self.render(data)


_default_router = OutputRouter()
_current_router: contextvars.ContextVar[OutputRouter] = contextvars.ContextVar(
    "seclab_output_router", default=_default_router
)


def get_output_router() -> OutputRouter:
    """Return the output router for the current context."""
    return _current_router.get()


def use_output_router(router: OutputRouter) -> contextvars.Token:
    """Install *router* as the current-context router; returns a reset token."""
    return _current_router.set(router)


async def flush_async_output(task_id: str, label: str | None = None) -> None:
    """Flush buffered async output for *task_id* to the console.

    When *label* is provided it replaces the default ``async task: <id>``
    heading, letting multi-model runs tag each block with its model name.
    """
    await get_output_router().flush(task_id, label=label)


async def render_model_output(
    data: str, log: bool = True, async_task: bool = False, task_id: str | None = None
) -> None:
    """Print model output to the console, optionally buffering for async tasks."""
    await get_output_router().render(data, log=log, async_task=async_task, task_id=task_id)
