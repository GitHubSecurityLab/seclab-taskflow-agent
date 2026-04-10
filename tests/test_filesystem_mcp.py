# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the filesystem MCP server (read_file and list_directory)."""

from __future__ import annotations

import os

import pytest

from seclab_taskflow_agent.mcp_servers.filesystem import filesystem


@pytest.fixture(autouse=True)
def _patch_base_dir(tmp_path, monkeypatch):
    """Point BASE_DIR at a temporary directory for every test."""
    monkeypatch.setattr(filesystem, "BASE_DIR", str(tmp_path))


@pytest.fixture
def _sample_file(tmp_path):
    """Create a sample file with 600 numbered lines."""
    p = tmp_path / "big.txt"
    p.write_text("".join(f"line {i}\n" for i in range(1, 601)))
    return p


@pytest.fixture
def _small_file(tmp_path):
    """Create a small 3-line file."""
    p = tmp_path / "small.txt"
    p.write_text("alpha\nbeta\ngamma\n")
    return p


# ===================================================================
# read_file
# ===================================================================


@pytest.mark.usefixtures("_sample_file")
class TestReadFile:
    def test_default_returns_first_500_lines_unchanged(self):
        result = filesystem.read_file("big.txt")
        lines = result.splitlines()
        assert len(lines) == 500
        assert lines[0] == "line 1"
        assert lines[499] == "line 500"

    def test_start_line_offsets_correctly(self):
        result = filesystem.read_file("big.txt", max_lines=3, start_line=100)
        lines = result.splitlines()
        assert len(lines) == 3
        assert lines[0] == "line 100"
        assert lines[2] == "line 102"

    def test_start_line_past_eof_returns_empty(self):
        result = filesystem.read_file("big.txt", start_line=9999)
        assert result == ""


@pytest.mark.usefixtures("_small_file")
class TestReadFileSmall:
    def test_line_numbers_format(self):
        result = filesystem.read_file("small.txt", line_numbers=True)
        lines = result.splitlines()
        assert lines[0] == "1: alpha"
        assert lines[1] == "2: beta"
        assert lines[2] == "3: gamma"

    def test_summary_shows_total_and_range(self):
        result = filesystem.read_file("small.txt", include_summary=True)
        assert "3 total lines" in result
        assert "showing 1-3" in result

    def test_summary_off_by_default(self):
        result = filesystem.read_file("small.txt")
        assert "total lines" not in result

    def test_summary_past_eof_explicit_message(self):
        result = filesystem.read_file("small.txt", start_line=999, include_summary=True)
        assert "no lines in range" in result

    def test_start_line_zero_clamped(self):
        result = filesystem.read_file("small.txt", start_line=0, line_numbers=True)
        lines = result.splitlines()
        assert lines[0] == "1: alpha"

    def test_backward_compatible_existing_calls(self):
        result = filesystem.read_file("small.txt", max_lines=500)
        assert "alpha" in result
        assert "gamma" in result


class TestReadFileErrors:
    def test_file_not_found(self):
        result = filesystem.read_file("nonexistent.txt")
        assert "Error" in result
        assert "not found" in result


# ===================================================================
# Path traversal
# ===================================================================


class TestPathTraversal:
    def test_dotdot_traversal_blocked(self):
        result = filesystem.read_file("../../etc/passwd")
        assert "path traversal not allowed" in result

    def test_sibling_prefix_escape_blocked(self, tmp_path):
        sibling = tmp_path.parent / (tmp_path.name + "-sibling")
        sibling.mkdir(exist_ok=True)
        secret = sibling / "secret.txt"
        secret.write_text("secret")
        rel = os.path.relpath(str(secret), str(tmp_path))
        result = filesystem.read_file(rel)
        assert "path traversal not allowed" in result

    def test_symlink_escape_blocked(self, tmp_path):
        external = tmp_path.parent / "external_target"
        external.mkdir(exist_ok=True)
        (external / "data.txt").write_text("external data")
        link = tmp_path / "escape_link"
        try:
            os.symlink(str(external), str(link))
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform")
        result = filesystem.read_file("escape_link/data.txt")
        assert "path traversal not allowed" in result

    def test_valid_subdirectory_allowed(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "file.txt").write_text("hello")
        result = filesystem.read_file("subdir/file.txt")
        assert result.strip() == "hello"

    def test_list_directory_traversal_blocked(self):
        result = filesystem.list_directory("../../etc")
        assert "path traversal not allowed" in result

    def test_list_directory_valid(self, tmp_path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        result = filesystem.list_directory(".")
        assert "a.txt" in result
        assert "b.txt" in result
