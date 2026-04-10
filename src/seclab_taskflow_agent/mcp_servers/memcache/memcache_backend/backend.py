# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

from typing import Any


class Backend:
    def __init__(self, memcache_state_dir: str):
        self.memcache_state_dir = memcache_state_dir

    def set_state(self, key: str, value: Any) -> str:
        pass

    def get_state(self, key: str) -> Any:
        pass

    def add_state(self, key: str, value: Any) -> str:
        pass

    def list_keys(self) -> str:
        pass

    def clear_cache(self) -> str:
        pass

    def snapshot_state(self) -> dict[str, Any]:
        """Return all keys with their merged values as a dict. Safe to call externally."""
        return {}
