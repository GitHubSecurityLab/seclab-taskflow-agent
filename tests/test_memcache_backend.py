# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for memcache backend snapshot_state and append_log."""

from __future__ import annotations

import threading

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


class TestAppendLogSqlite:
    def test_creates_list_on_first_append(self, tmp_path):
        b = SqliteBackend(str(tmp_path))
        b.append_log("findings", "xss found")
        entries = b.get_log("findings")
        assert len(entries) == 1
        assert entries[0]["data"] == "xss found"

    def test_appends_to_existing_list(self, tmp_path):
        b = SqliteBackend(str(tmp_path))
        b.append_log("findings", "xss")
        b.append_log("findings", "sqli")
        entries = b.get_log("findings")
        assert len(entries) == 2
        assert entries[0]["data"] == "xss"
        assert entries[1]["data"] == "sqli"

    def test_entries_have_timestamps(self, tmp_path):
        b = SqliteBackend(str(tmp_path))
        b.append_log("k", "v")
        entries = b.get_log("k")
        assert "_ts" in entries[0]

    def test_get_log_returns_all_entries_in_order(self, tmp_path):
        b = SqliteBackend(str(tmp_path))
        for i in range(10):
            b.append_log("ordered", f"entry-{i}")
        entries = b.get_log("ordered")
        assert len(entries) == 10
        for i, entry in enumerate(entries):
            assert entry["data"] == f"entry-{i}"

    def test_get_log_empty_key_returns_empty_list(self, tmp_path):
        b = SqliteBackend(str(tmp_path))
        assert b.get_log("nonexistent") == []

    def test_multiple_sequential_appends_no_data_loss(self, tmp_path):
        b = SqliteBackend(str(tmp_path))
        n = 50
        for i in range(n):
            b.append_log("bulk", f"item-{i}")
        assert len(b.get_log("bulk")) == n

    def test_concurrent_appends_no_data_loss(self, tmp_path):
        b = SqliteBackend(str(tmp_path))
        n_threads = 5
        n_per_thread = 20

        def worker(thread_id):
            for i in range(n_per_thread):
                b.append_log("concurrent", f"t{thread_id}-{i}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = b.get_log("concurrent")
        assert len(entries) == n_threads * n_per_thread


class TestAppendLogDictFile:
    def test_creates_list_on_first_append(self, tmp_path):
        b = MemcacheDictionaryFileBackend(str(tmp_path))
        b.append_log("findings", "xss found")
        entries = b.get_log("findings")
        assert len(entries) == 1
        assert entries[0]["data"] == "xss found"

    def test_appends_to_existing_list(self, tmp_path):
        b = MemcacheDictionaryFileBackend(str(tmp_path))
        b.append_log("findings", "xss")
        b.append_log("findings", "sqli")
        entries = b.get_log("findings")
        assert len(entries) == 2

    def test_get_log_empty_key_returns_empty_list(self, tmp_path):
        b = MemcacheDictionaryFileBackend(str(tmp_path))
        assert b.get_log("nonexistent") == []


class TestSetStateReturnFix:
    def test_set_state_return_contains_key_name(self, tmp_path):
        b = SqliteBackend(str(tmp_path))
        result = b.set_state("my_key", "value")
        assert "my_key" in result
        # Should NOT be a literal f-string
        assert "{key}" not in result


class TestSetStateUnchanged:
    def test_set_state_replaces_value(self, tmp_path):
        b = SqliteBackend(str(tmp_path))
        b.set_state("k", "old")
        b.set_state("k", "new")
        assert b.get_state("k") == "new"
