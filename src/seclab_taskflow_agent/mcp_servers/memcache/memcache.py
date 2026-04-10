# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

import json
import logging
import os
from typing import Any

# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP  # move to FastMCP 2.0

from seclab_taskflow_agent.path_utils import log_file_name, mcp_data_dir

from .memcache_backend.dictionary_file import MemcacheDictionaryFileBackend
from .memcache_backend.sqlite import SqliteBackend

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_file_name("mcp_memcache.log"),
    filemode="a",
)

mcp = FastMCP("Memcache")

backends = {
    "dictionary_file": MemcacheDictionaryFileBackend,
    "sqlite": SqliteBackend,
}

MEMORY = mcp_data_dir("seclab-taskflow-agent", "memcache", "MEMCACHE_STATE_DIR")
BACKEND = os.getenv("MEMCACHE_BACKEND", default="sqlite")

backend = backends.get(BACKEND)(str(MEMORY))


@mcp.tool()
def memcache_set_state(key: str, value: Any) -> str:
    """Set or override a value for a key into the memory cache. This acts as your memory."""
    return backend.set_state(key, value)


@mcp.tool()
def memcache_get_state(key: str) -> str:
    """Get a value for a key from the memory cache. Returned values are JSON serialized object strings."""
    return json.dumps(backend.get_state(key))


@mcp.tool()
def memcache_list_keys() -> str:
    """List all available keys in your memory cache."""
    return backend.list_keys()


@mcp.tool()
def memcache_get_all_entries() -> str:
    """Get all entries in your memory cache. Returned values are JSON serialized object strings."""
    return json.dumps(backend.get_all_entries())


@mcp.tool()
def memcache_add_state(key: str, value: Any) -> str:
    """Add to the existing value for an existing key in your memory cache. Supports lists and strings."""
    return backend.add_state(key, value)


@mcp.tool()
def memcache_delete_state(key: str) -> str:
    """Delete a key from the memory cache."""
    return backend.delete_state(key)


@mcp.tool()
def memcache_clear_cache():
    """Clear the memory cache, invalidating all stored key value pairs."""
    return backend.clear_cache()


@mcp.tool()
def memcache_append_log(key: str, entry: Any) -> str:
    """Append a timestamped entry to an append-only log under the given key.
    Use this for findings or notes that should accumulate, not be replaced.
    Retrieve entries with memcache_get_log(key)."""
    return backend.append_log(key, entry)


@mcp.tool()
def memcache_get_log(key: str) -> str:
    """Retrieve all entries from an append-only log created by memcache_append_log."""
    import json as _json

    entries = backend.get_log(key)
    return _json.dumps(entries, indent=2)


if __name__ == "__main__":
    mcp.run(show_banner=False)
