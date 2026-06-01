# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""MCP transport-layer implementations.

Provides thread-based local MCP server process management and specialised
stdio wrappers that work around asyncio and JSON-RPC edge cases.

Classes:
    StreamableMCPThread: Manages a local streamable MCP server process.
    AsyncDebugMCPServerStdio: Debug wrapper that isolates the asyncio loop.
    ReconnectingMCPServerStdio: Reconnecting wrapper for flaky stdio I/O.
"""

from __future__ import annotations

__all__ = [
    "AsyncDebugMCPServerStdio",
    "ReconnectingMCPServerStdio",
    "StreamableMCPThread",
    "_filtered_env",
]

import asyncio
import logging
import os
import signal
import socket
import subprocess
import time
from collections.abc import Callable
from threading import Event, Thread
from typing import IO, Any
from urllib.parse import urlparse

from agents.mcp import MCPServerStdio

# Exit codes that are considered normal termination.
_EXPECTED_EXIT_CODES: frozenset[int] = frozenset({0, -signal.SIGTERM})


def _filtered_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` with denied variables removed.

    Set ``TASKFLOW_ENV_DENYLIST`` to a comma-separated list of variable
    names that should not be forwarded to MCP server subprocesses.
    Toolbox-level ``env:`` declarations in YAML still inject what each
    server explicitly needs.
    """
    denylist_raw = os.environ.get("TASKFLOW_ENV_DENYLIST", "")
    if not denylist_raw:
        return os.environ.copy()
    denied = {k.strip() for k in denylist_raw.split(",") if k.strip()}
    return {k: v for k, v in os.environ.items() if k not in denied}


class StreamableMCPThread(Thread):
    """Thread that manages a local streamable MCP server subprocess.

    The thread starts the server, reads its stdout/stderr via callbacks,
    and terminates the process when :meth:`stop` is called.

    Args:
        cmd: Command-line tokens to launch the server.
        url: URL the server will listen on (used for connection probes).
        on_output: Callback invoked with each stdout line.
        on_error: Callback invoked with each stderr line.
        poll_interval: Seconds between process-alive checks.
        env: Extra environment variables merged into the current env.
    """

    def __init__(
        self,
        cmd: list[str],
        url: str = "",
        on_output: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        poll_interval: float = 0.5,
        env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.url: str = url
        self.cmd: list[str] = cmd
        self.on_output: Callable[[str], None] | None = on_output
        self.on_error: Callable[[str], None] | None = on_error
        self.poll_interval: float = poll_interval
        self.env: dict[str, str] = _filtered_env()
        if env:
            self.env.update(env)
        self._stop_event: Event = Event()
        self.process: subprocess.Popen[str] | None = None
        self.exit_code: int | None = None
        self.exception: BaseException | None = None

    async def async_wait_for_connection(
        self, timeout: float = 30.0, poll_interval: float = 0.5
    ) -> None:
        """Asynchronously poll until the server accepts TCP connections.

        Args:
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between connection attempts.

        Raises:
            ValueError: If *url* is missing host or port.
            TimeoutError: If the server is not reachable within *timeout*.
        """
        parsed = urlparse(self.url)
        host = parsed.hostname
        port = parsed.port
        if host is None or port is None:
            raise ValueError(f"URL must include a host and port: {self.url}")
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            try:
                reader, writer = await asyncio.open_connection(host, port)
                writer.close()
                await writer.wait_closed()
                return
            except (OSError, ConnectionRefusedError):
                if asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError(f"Could not connect to {host}:{port} after {timeout} seconds")
                await asyncio.sleep(poll_interval)

    def wait_for_connection(
        self, timeout: float = 30.0, poll_interval: float = 0.5
    ) -> None:
        """Synchronously poll until the server accepts TCP connections.

        Args:
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between connection attempts.

        Raises:
            ValueError: If *url* is missing host or port.
            TimeoutError: If the server is not reachable within *timeout*.
        """
        parsed = urlparse(self.url)
        host = parsed.hostname
        port = parsed.port
        if host is None or port is None:
            raise ValueError(f"URL must include a host and port: {self.url}")
        deadline = time.time() + timeout
        while True:
            try:
                with socket.create_connection((host, port), timeout=2):
                    return
            except OSError:
                if time.time() > deadline:
                    raise TimeoutError(f"Could not connect to {host}:{port} after {timeout} seconds")
                time.sleep(poll_interval)

    def run(self) -> None:
        """Execute the subprocess and monitor it until stopped."""
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=self.env,
            )

            stdout_thread = Thread(target=self._read_stream, args=(self.process.stdout, self.on_output))
            stderr_thread = Thread(target=self._read_stream, args=(self.process.stderr, self.on_error))
            stdout_thread.start()
            stderr_thread.start()

            while self.process.poll() is None and not self._stop_event.is_set():
                time.sleep(self.poll_interval)

            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait()
            self.exit_code = self.process.returncode

            stdout_thread.join()
            stderr_thread.join()

            if self.exit_code not in _EXPECTED_EXIT_CODES:
                self.exception = subprocess.CalledProcessError(self.exit_code, self.cmd)

        except Exception as e:
            self.exception = e

    def _read_stream(
        self, stream: IO[str] | None, callback: Callable[[str], None] | None
    ) -> None:
        """Drain *stream* line-by-line, forwarding to *callback*."""
        if stream is None or callback is None:
            return
        for line in iter(stream.readline, ""):
            callback(line.rstrip("\n"))
        stream.close()

    def stop(self) -> None:
        """Request the subprocess to terminate."""
        self._stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def is_running(self) -> bool:
        """Return whether the subprocess is still alive."""
        return self.process is not None and self.process.poll() is None

    def join_and_raise(self, timeout: float | None = None) -> None:
        """Join the thread and re-raise any captured exception.

        Args:
            timeout: Maximum seconds to wait for the thread to finish.

        Raises:
            RuntimeError: If the thread is still alive after *timeout*.
        """
        self.join(timeout)
        if self.is_alive():
            msg = "Process thread did not exit within timeout."
            raise RuntimeError(msg)
        if self.exception is not None:
            raise self.exception


class AsyncDebugMCPServerStdio(MCPServerStdio):
    """Debug wrapper that runs MCP stdio operations on a dedicated asyncio loop.

    Useful for diagnosing event-loop conflicts when the main loop is shared
    with other coroutines.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        class _AsyncLoopThread(Thread):
            """Daemon thread owning an isolated asyncio event loop."""

            def __init__(self) -> None:
                super().__init__(daemon=True)
                self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()

            def run(self) -> None:
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()

        self.t = _AsyncLoopThread()
        self.t.start()
        self.lock: asyncio.Lock = asyncio.Lock()

    async def connect(self, *args: Any, **kwargs: Any) -> Any:
        """Connect via the dedicated loop."""
        return asyncio.run_coroutine_threadsafe(super().connect(*args, **kwargs), self.t.loop).result()

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        """List tools via the dedicated loop."""
        return asyncio.run_coroutine_threadsafe(super().list_tools(*args, **kwargs), self.t.loop).result()

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        """Call a tool via the dedicated loop (serialised with a lock)."""
        async with self.lock:
            return asyncio.run_coroutine_threadsafe(super().call_tool(*args, **kwargs), self.t.loop).result()

    async def cleanup(self, *args: Any, **kwargs: Any) -> None:
        """Clean up and shut down the dedicated loop."""
        try:
            asyncio.run_coroutine_threadsafe(super().cleanup(*args, **kwargs), self.t.loop).result()
        except asyncio.CancelledError:
            pass  # Swallow cancellation during cleanup shutdown
        finally:
            self.t.loop.stop()
            self.t.join()


class ReconnectingMCPServerStdio(MCPServerStdio):
    """Stdio wrapper that reconnects before every tool operation.

    Works around buggy JSON-RPC stdio behaviour in FastMCP 1.0 where
    long-running, high-volume processes miss I/O and results never arrive
    on the client side.  Enable via ``reconnecting: true`` in your
    toolbox config.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reconnecting_lock: asyncio.Lock = asyncio.Lock()

    async def connect(self) -> None:
        """No-op — connections are opened per-call."""
        logging.debug("Ignoring mcp connect request on purpose")

    async def cleanup(self) -> None:
        """No-op — cleanup happens per-call."""
        logging.debug("Ignoring mcp cleanup request on purpose")

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        """Connect, list tools, then disconnect."""
        async with self.reconnecting_lock:
            await super().connect()
            try:
                result = await super().list_tools(*args, **kwargs)
            finally:
                await super().cleanup()
            return result

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        """Connect, call tool, then disconnect."""
        logging.debug("Using reconnecting call_tool for stdio mcp")
        async with self.reconnecting_lock:
            await super().connect()
            try:
                result = await super().call_tool(*args, **kwargs)
            finally:
                await super().cleanup()
            return result
