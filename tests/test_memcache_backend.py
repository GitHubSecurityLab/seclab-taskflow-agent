# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for memcache backend snapshot_state and append_log."""

from __future__ import annotations

from seclab_taskflow_agent.mcp_servers.memcache.memcache_backend.dictionary_file import (
    MemcacheDictionaryFileBackend,
)
from seclab_taskflow_agent.mcp_servers.memcache.memcache_backend.sqlite import SqliteBackend


class TestSnapshotStateSqlite:
    def test_snapshot_returns_all_keys(self, tmp_path):
        b = SqliteBackend(str(tmp_path))
        b.set_state("key1", "val1")
        b.set_state("key2", [1, 2, 3])
        snap = b.snapshot_state()
        assert snap["key1"] == "val1"
        assert snap["key2"] == [1, 2, 3]

    def test_snapshot_empty_db(self, tmp_path):
        b = SqliteBackend(str(tmp_path))
        assert b.snapshot_state() == {}


class TestSnapshotStateDictFile:
    def test_snapshot_returns_deep_copy(self, tmp_path):
        b = MemcacheDictionaryFileBackend(str(tmp_path))
        b.set_state("a", {"nested": True})
        snap = b.snapshot_state()
        assert snap["a"] == {"nested": True}
        # Mutating nested values in the snapshot shouldn't affect the backend
        snap["a"]["nested"] = False
        assert b.get_state("a") == {"nested": True}

    def test_snapshot_empty(self, tmp_path):
        b = MemcacheDictionaryFileBackend(str(tmp_path))
        assert b.snapshot_state() == {}
