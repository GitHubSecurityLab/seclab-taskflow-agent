# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

import logging
import os

from fastmcp import FastMCP

from seclab_taskflow_agent.path_utils import log_file_name

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_file_name("mcp_filesystem.log"),
    filemode="a",
)

mcp = FastMCP("Filesystem")

BASE_DIR = os.getenv("FILESYSTEM_BASE_DIR", os.getcwd())


@mcp.tool()
def list_directory(path: str = ".") -> str:
    """List files and directories relative to the repo root."""
    target = os.path.realpath(os.path.join(BASE_DIR, path))
    base = os.path.realpath(BASE_DIR)
    if not target.startswith(base + os.sep) and target != base:
        return "Error: path traversal not allowed"
    if not os.path.isdir(target):
        return f"Error: {path} is not a directory"
    try:
        entries = sorted(os.listdir(target))
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"Error listing {path}: {e}"


@mcp.tool()
def read_file(
    path: str,
    max_lines: int = 500,
    start_line: int = 1,
    line_numbers: bool = False,
    include_summary: bool = False,
) -> str:
    """Read a file's contents relative to the repo root.
    Returns up to max_lines lines starting from start_line (1-indexed).
    When line_numbers is True, each line is prefixed with its number.
    When include_summary is True, appends a footer with total line count and range."""
    target = os.path.realpath(os.path.join(BASE_DIR, path))
    base = os.path.realpath(BASE_DIR)
    if not target.startswith(base + os.sep) and target != base:
        return "Error: path traversal not allowed"
    if not os.path.isfile(target):
        return f"Error: {path} not found"
    try:
        with open(target, errors="replace") as f:
            all_lines = f.readlines()
        total = len(all_lines)
        start_idx = max(0, start_line - 1)
        selected = all_lines[start_idx : start_idx + max_lines]

        output = (
            [f"{start_idx + i + 1}: {ln}" for i, ln in enumerate(selected)]
            if line_numbers
            else list(selected)
        )

        result = "".join(output)
        if include_summary:
            if not selected:
                result += f"\n--- {total} total lines, no lines in range {start_idx + 1}+ ---"
            else:
                actual_end = start_idx + len(selected)
                result += f"\n--- {total} total lines, showing {start_idx + 1}-{actual_end} ---"
        return result
    except Exception as e:
        return f"Error reading {path}: {e}"


if __name__ == "__main__":
    mcp.run(show_banner=False)
