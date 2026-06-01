# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Shell command execution utilities."""

import logging
import subprocess
import tempfile

from mcp.types import CallToolResult, TextContent

__all__ = ["shell_command_to_string", "shell_exec_with_temporary_file", "shell_tool_call"]


def shell_command_to_string(cmd: list[str]) -> str:
    """Execute a shell command and return its stdout.

    Raises:
        RuntimeError: If the command exits with a non-zero return code.
    """
    logging.info("Executing: %s", cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    stdout, stderr = p.communicate()
    p.wait()
    if p.returncode:
        raise RuntimeError(f"Command {cmd} failed: {stderr}")
    return stdout


def shell_exec_with_temporary_file(script: str, shell: str = "bash") -> str:
    """Write *script* to a temp file and execute it with the given shell."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
        temp_file.write(script)
        temp_file.flush()
        result = shell_command_to_string([shell, temp_file.name])
        return result


def shell_tool_call(run: str) -> CallToolResult:
    """Execute a shell script and return the output as a CallToolResult."""
    stdout = shell_exec_with_temporary_file(run)
    # this allows e.g. shell based jq output to become available for repeat prompts
    result = CallToolResult(content=[TextContent(type="text", text=stdout, annotations=None, meta=None)])
    return result
