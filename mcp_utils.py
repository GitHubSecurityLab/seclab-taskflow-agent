import logging
import asyncio
from threading import Thread, Event
import json
import subprocess
from typing import Optional, Callable
import shutil
import time
import os
import socket
import signal
from urllib.parse import urlparse

from mcp.types import CallToolResult, TextContent
from agents.mcp import MCPServerStdio

from .env_utils import swap_env

DEFAULT_MCP_CLIENT_SESSION_TIMEOUT = 120

# A process management class for running in-process MCP streamable servers
class StreamableMCPThread(Thread):
    """Process management for local streamable MCP servers"""
    def __init__(
            self,
            cmd,
            url: str = '',
            on_output: Optional[Callable[[str], None]] = None,
            on_error: Optional[Callable[[str], None]] = None,
            poll_interval: float = 0.5,
            env: Optional[dict[str, str]] = None
    ):
        super().__init__(daemon=True)
        self.url = url
        self.cmd = cmd
        self.on_output = on_output
        self.on_error = on_error
        self.poll_interval = poll_interval
        self.env = os.environ.copy() # XXX: potential for environment leak to MCP
        self.env.update(env)
        self._stop_event = Event()
        self.process = None
        self.exit_code = None
        self.exception: Optional[BaseException] = None

    async def async_wait_for_connection(self, timeout=30.0, poll_interval=0.5):
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
                return  # Success
            except (OSError, ConnectionRefusedError):
                if asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError(f"Could not connect to {host}:{port} after {timeout} seconds")
                await asyncio.sleep(poll_interval)

    def wait_for_connection(self, timeout=30.0, poll_interval=0.5):
        parsed = urlparse(self.url)
        host = parsed.hostname
        port = parsed.port
        if host is None or port is None:
            raise ValueError(f"URL must include a host and port: {self.url}")
        deadline = time.time() + timeout
        while True:
            try:
                with socket.create_connection((host, port), timeout=2):
                    return  # Success
            except OSError:
                if time.time() > deadline:
                    raise TimeoutError(f"Could not connect to {host}:{port} after {timeout} seconds")
                time.sleep(poll_interval)

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=self.env
            )

            stdout_thread = Thread(target=self._read_stream, args=(self.process.stdout, self.on_output))
            stderr_thread = Thread(target=self._read_stream, args=(self.process.stderr, self.on_error))
            stdout_thread.start()
            stderr_thread.start()

            while self.process.poll() is None and not self._stop_event.is_set():
                time.sleep(self.poll_interval)

            # Process ended or stop requested
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait()
            self.exit_code = self.process.returncode

            stdout_thread.join()
            stderr_thread.join()

            # sigterm (-15) is expected
            if self.exit_code not in [0, -15]:
                self.exception = subprocess.CalledProcessError(
                    self.exit_code, self.cmd
                )

        except BaseException as e:
            self.exception = e

    def _read_stream(self, stream, callback):
        if stream is None or callback is None:
            return
        for line in iter(stream.readline, ''):
            callback(line.rstrip('\n'))
        stream.close()

    def stop(self):
        self._stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def is_running(self):
        return self.process and self.process.poll() is None

    def join_and_raise(self, timeout: Optional[float] = None):
        self.join(timeout)
        if self.is_alive():
            raise RuntimeError("Process thread did not exit within timeout.")
        if self.exception is not None:
            raise self.exception

# used for debugging asyncio event loop issues in mcp stdio servers
# lifts the asyncio event loop in use to a dedicated threaded loop
class AsyncDebugMCPServerStdio(MCPServerStdio):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class AsyncLoopThread(Thread):
            def __init__(self):
                super().__init__(daemon=True)
                self.loop = asyncio.new_event_loop()
            def run(self):
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()
        self.t = AsyncLoopThread()
        self.t.start()
        self.lock = asyncio.Lock()

    async def connect(self, *args, **kwargs):
        return asyncio.run_coroutine_threadsafe(
            super().connect(*args, **kwargs),
            self.t.loop).result()

    async def list_tools(self, *args, **kwargs):
        return asyncio.run_coroutine_threadsafe(
            super().list_tools(*args, **kwargs),
            self.t.loop).result()

    async def call_tool(self, *args, **kwargs):
        async with self.lock:
            return asyncio.run_coroutine_threadsafe(
                super().call_tool(*args, **kwargs),
                self.t.loop).result()

    async def cleanup(self, *args, **kwargs):
        try:
            asyncio.run_coroutine_threadsafe(
                super().cleanup(*args, **kwargs),
                self.t.loop).result()
        except asyncio.CancelledError:
            pass
        finally:
            self.t.loop.stop()
            self.t.join()


# a hack class that works around buggy jsonrpc stdio behavior in FastMCP 1.0
# long running high volume processes tend to get confused and miss i/o
# if you're seeing behavior where your mcp server tool call completes
# but the results never arrive to to the mcp client side, try and set
# reconnecting: true in your toolbox config
class ReconnectingMCPServerStdio(MCPServerStdio):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconnecting_lock = asyncio.Lock()

    async def connect(self):
        logging.debug("Ignoring mcp connect request on purpose")

    async def cleanup(self):
        logging.debug("Ignoring mcp cleanup request on purpose")

    async def list_tools(self, *args, **kwargs):
        async with self.reconnecting_lock:
            await super().connect()
            try:
                result = await super().list_tools(*args, **kwargs)
            finally:
                await super().cleanup()
            return result

    async def call_tool(self, *args, **kwargs):
        logging.debug("Using reconnecting call_tool for stdio mcp")
        async with self.reconnecting_lock:
            await super().connect()
            try:
                result = await super().call_tool(*args, **kwargs)
            finally:
                await super().cleanup()
            return result

class MCPNamespaceWrap:
    """An MCP client object wrapper that provides us with namespace control"""
    def __init__(self, confirms, obj):
        self.confirms = confirms
        self._obj = obj
        self.namespace = f"{obj.name.upper().replace(' ', '_')}_"

    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        if callable(attr):
            match name:
                case 'call_tool':
                    return self.call_tool
                case 'list_tools':
                    return self.list_tools
                case _:
                    return attr
        return attr

    async def list_tools(self, *args, **kwargs):
        result = await self._obj.list_tools(*args, **kwargs)
        namespaced_tools = []
        for tool in result:
            tool_copy = tool.copy()
            tool_copy.name = f"{self.namespace}{tool.name}"
            namespaced_tools.append(tool_copy)
        return namespaced_tools

    def confirm_tool(self, tool_name, args):
        while True:
            yn = input(f"** 🤖❗ Allow tool call?: {tool_name}({','.join([json.dumps(arg) for arg in args])}) (yes/no): ")
            if yn in ["yes", "y"]:
                return True
            elif yn in ["no", "n"]:
                return False

    async def call_tool(self, *args, **kwargs):
        _args = list(args)
        tool_name = _args[0]
        tool_name = tool_name.removeprefix(self.namespace)
        # to run headless, just make confirms an empty list
        if self.confirms and tool_name in self.confirms:
            if not self.confirm_tool(tool_name, _args[1:]):
                result = CallToolResult(
                    content=[
                        TextContent(
                            type='text',
                            text='Tool call not allowed.',
                            annotations=None,
                            meta=None)]
                )
                return result
        _args[0] = tool_name
        args = tuple(_args)
        result = await self._obj.call_tool(*args, **kwargs)
        return result

def mcp_client_params(available_toolboxes: dict, requested_toolboxes: list):
    """Return all the data needed to initialize an mcp server client"""
    client_params = {}
    for tb in requested_toolboxes:
        if tb not in available_toolboxes:
            e = f"Task requested non-existent toolbox {tb}!"
            logging.critical(e)
            raise ValueError(e)
        else:
            kind = available_toolboxes[tb]['server_params'].get('kind')
            reconnecting = available_toolboxes[tb]['server_params'].get('reconnecting', False)
            server_params = {'kind': kind, 'reconnecting': reconnecting}
            match kind:
                case 'stdio':
                    env = available_toolboxes[tb]['server_params'].get('env')
                    args = available_toolboxes[tb]['server_params'].get('args')
                    logging.debug(f"Initializing toolbox: {tb}\nargs:\n{args }\nenv:\n{env}\n")
                    if env and isinstance(env, dict):
                        for k, v in dict(env).items():
                            try:
                                env[k] = swap_env(v)
                            except LookupError as e:
                                logging.critical(e)
                                logging.info("Assuming toolbox has default configuration available")
                                del env[k]
                                pass
                    logging.debug(f"Tool call environment: {env}")
                    if args and isinstance(args, list):
                        for i, v in enumerate(args):
                            args[i] = swap_env(v)
                    logging.debug(f"Tool call args: {args}")
                    server_params['command'] = available_toolboxes[tb]['server_params'].get('command')
                    server_params['args'] = args
                    server_params['env'] = env
                # XXX: SSE is deprecated in the MCP spec, but keep it around for now
                case 'sse':
                    headers = available_toolboxes[tb]['server_params'].get('headers')
                    # support {{ env SOMETHING }} for header values as well for e.g. tokens
                    if headers and isinstance(headers, dict):
                        for k, v in headers.items():
                            headers[k] = swap_env(v)
                    optional_headers = available_toolboxes[tb]['server_params'].get('optional_headers')
                    # support {{ env SOMETHING }} for header values as well for e.g. tokens
                    if optional_headers and isinstance(optional_headers, dict):
                        for k, v in dict(optional_headers).items():
                            try:
                                optional_headers[k] = swap_env(v)
                            except LookupError as e:
                                del optional_headers[k]
                    if isinstance(headers, dict):
                        if isinstance(optional_headers, dict):
                            headers.update(optional_headers)
                    elif isinstance(optional_headers, dict):
                        headers = optional_headers
                    # if None will default to float(5) in client code
                    timeout = available_toolboxes[tb]['server_params'].get('timeout')
                    server_params['url'] = available_toolboxes[tb]['server_params'].get('url')
                    server_params['headers'] = headers
                    server_params['timeout'] = timeout
                # for more involved local MCP servers, jsonrpc over stdio seems less than reliable
                # as an alternative you can configure local toolboxes to use the streamable transport
                # but still be started/stopped on demand similar to stdio mcp servers
                # all it requires is a streamable config that also has cmd/args/env set
                case 'streamable':
                    headers = available_toolboxes[tb]['server_params'].get('headers')
                    # support {{ env SOMETHING }} for header values as well for e.g. tokens
                    if headers and isinstance(headers, dict):
                        for k, v in headers.items():
                            headers[k] = swap_env(v)
                    optional_headers = available_toolboxes[tb]['server_params'].get('optional_headers')
                    # support {{ env SOMETHING }} for header values as well for e.g. tokens
                    if optional_headers and isinstance(optional_headers, dict):
                        for k, v in dict(optional_headers).items():
                            try:
                                optional_headers[k] = swap_env(v)
                            except LookupError as e:
                                del optional_headers[k]
                    if isinstance(headers, dict):
                        if isinstance(optional_headers, dict):
                            headers.update(optional_headers)
                    elif isinstance(optional_headers, dict):
                        headers = optional_headers
                    # if None will default to float(5) in client code
                    timeout = available_toolboxes[tb]['server_params'].get('timeout')
                    server_params['url'] = available_toolboxes[tb]['server_params'].get('url')
                    server_params['headers'] = headers
                    server_params['timeout'] = timeout
                    # if command/args/env is set, we also need to start this MCP server ourselves
                    # this way we can use the streamable transport for MCP servers that get fussy
                    # over stdio jsonrpc polling
                    env = available_toolboxes[tb]['server_params'].get('env')
                    args = available_toolboxes[tb]['server_params'].get('args')
                    cmd = available_toolboxes[tb]['server_params'].get('command')
                    if cmd is not None:
                        logging.debug(f"Initializing streamable toolbox: {tb}\nargs:\n{args }\nenv:\n{env}\n")
                        exe = shutil.which(cmd)
                        if exe is None:
                            raise FileNotFoundError(f"Could not resolve path to {cmd}")
                        start_cmd = [exe]
                        if args is not None and isinstance(args, list):
                            for i, v in enumerate(args):
                                args[i] = swap_env(v)
                            start_cmd += args
                        server_params['command'] = start_cmd
                        if env is not None and isinstance(env, dict):
                            for k, v in dict(env).items():
                                try:
                                    env[k] = swap_env(v)
                                except LookupError as e:
                                    logging.critical(e)
                                    logging.info("Assuming toolbox has default configuration available")
                                    del env[k]
                                    pass
                        server_params['env'] = env
                case _:
                    raise ValueError(f"Unsupported MCP transport {kind}")
            confirms = available_toolboxes[tb].get('confirm', [])
            server_prompt = available_toolboxes[tb].get('server_prompt', '')
            client_session_timeout = float(available_toolboxes[tb].get('client_session_timeout', 0))
            client_params[tb] = (server_params, confirms, server_prompt, client_session_timeout)
    return client_params

def mcp_system_prompt(system_prompt: str, task: str,
                      tools: list[str] = [],
                      resources: list[str] = [],
                      resource_templates: list[str] = [],
                      important_guidelines: list[str] = [],
                      server_prompts: list[str] = []):
    """Return a well constructed system prompt"""
    prompt = """
{system_prompt}
""".format(system_prompt=system_prompt)

    if tools:
        prompt += """

# Available Tools

- {tools}
""".format(tools="\n- ".join(tools))

    if resources:
        prompt += """

# Available Resources

- {resources}
""".format(resources="\n- ".join(resources))

    if resource_templates:
        prompt += """

# Available Resource Templates

- {resource_templates}
""".format(resource_templates="\n- ".join(resource_templates))

    if important_guidelines:
        prompt += """

# Important Guidelines

- IMPORTANT: {guidelines}
""".format(guidelines="\n- IMPORTANT: ".join(important_guidelines))

    if server_prompts:
        prompt += """

# Additional Guidelines

{server_prompts}

""".format(server_prompts="\n\n".join(server_prompts))

    if task:
        prompt += """

# Primary Task to Complete

{task}

""".format(task=task)

    return prompt
