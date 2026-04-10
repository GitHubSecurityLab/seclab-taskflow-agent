# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
from typing import Any

from .backend import Backend


class MemcacheDictionaryFileBackend(Backend):
    """A simple dictionary file backend for a memory cache."""

    def __init__(self, path: str):
        super().__init__(path)
        self.memory = Path(self.memcache_state_dir) / Path("memory.json")
        self.memcache = {}

    def _ensure_memory(self):
        try:
            self.memory.parent.mkdir(exist_ok=True, parents=True)
            with open(self.memory, "x") as memory:
                memory.write(json.dumps(self.memcache))
                memory.flush()
        except FileExistsError:
            pass

    def _deflate_memory(self):
        self._ensure_memory()
        with open(self.memory, "w") as memory:
            memory.write(json.dumps(self.memcache))
            memory.flush()

    def _inflate_memory(self):
        self._ensure_memory()
        with open(self.memory) as memory:
            self.memcache = json.loads(memory.read())

    def with_memory(self, f):
        """Decorator to ensure memory is inflated before and deflated after the function call."""

        def wrapper(*args, **kwargs):
            self._inflate_memory()
            ret = f(*args, **kwargs)
            self._deflate_memory()
            return ret

        return wrapper

    def set_state(self, key, value):
        @self.with_memory
        def _set_state(key: str, value: Any) -> str:
            self.memcache[key] = value
            return f"Stored value in memory for `{key}`"

        return _set_state(key, value)

    def get_state(self, key):
        @self.with_memory
        def _get_state(key: str) -> Any:
            value = self.memcache.get(key, "")
            return value

        return _get_state(key)

    def delete_state(self, key):
        @self.with_memory
        def _delete_state(key: str) -> str:
            if key in self.memcache:
                del self.memcache[key]
                return f"Deleted key `{key}` from memory cache."
            return f"Key `{key}` not found in memory cache."

        return _delete_state(key)

    def get_all_entries(self):
        @self.with_memory
        def _get_all_entries() -> str:
            return [{"key": k, "value": v} for k, v in self.memcache.items()]

        return _get_all_entries()

    def add_state(self, key, value):
        @self.with_memory
        def _add_state(key: str, value: Any) -> str:
            existing = self.memcache.get(key)
            if type(existing) is type(value) and hasattr(existing, "__add__"):
                self.memcache[key] = existing + value
                return f"Updated and added to value in memory for key: `{key}`"
            if isinstance(existing, list):
                self.memcache[key].append(value)
                return f"Updated and added to value in memory for key: `{key}`"
            return f"Error: unsupported types for memcache add `{type(existing)} + {type(value)}` for key `{key}`"

        return _add_state(key, value)

    def list_keys(self):
        @self.with_memory
        def _list_keys() -> str:
            content = []
            content.append("IMPORTANT: your known memcache keys are now:\n")
            content += [f"- {key}" for key in self.memcache]
            return "\n".join(content)

        return _list_keys()

    def clear_cache(self):
        @self.with_memory
        def _clear_cache() -> str:
            self.memcache = {}
            return "Memory cache was cleared, all previous key lists are invalidated."

        return _clear_cache()

    def snapshot_state(self) -> dict[str, Any]:
        """Return a deep copy of the in-memory dictionary."""
        import copy

        self._inflate_memory()
        return copy.deepcopy(self.memcache)
