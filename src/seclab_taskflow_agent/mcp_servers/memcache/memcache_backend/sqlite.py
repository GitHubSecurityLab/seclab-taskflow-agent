# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from .backend import Backend
from .sql_models import Base, KeyValue


class SqliteBackend(Backend):
    def __init__(self, memcache_state_dir: str):
        super().__init__(memcache_state_dir)
        if not Path(self.memcache_state_dir).exists():
            db_dir = "sqlite://"
        else:
            db_dir = f"sqlite:///{os.path.abspath(self.memcache_state_dir)}/memory.db"
        self.engine = create_engine(db_dir, echo=False)
        Base.metadata.create_all(self.engine, tables=[KeyValue.__table__])

    def set_state(self, key: str, value: Any) -> str:
        with Session(self.engine) as session:
            session.query(KeyValue).filter_by(key=key).delete()
            kv = KeyValue(key=key, value=json.dumps(value))
            session.add(kv)
            session.commit()
        return f"Stored value in memory for `{key}`"

    def get_state(self, key: str) -> Any:
        with Session(self.engine) as session:
            values = session.query(KeyValue).filter_by(key=key).all()
        results = [v for v in values]
        if not results:
            return ""
        results.sort(key=lambda x: x.id)
        results = [json.loads(v.value) for v in results]
        if len(results) == 1:
            return results[0]
        existing = results[0]
        if isinstance(existing, list):
            for r in results[1:]:
                existing.append(r)
            return existing
        if hasattr(existing, "__add__"):
            try:
                for r in results[1:]:
                    existing += r
                return existing
            except TypeError:
                return results

    def add_state(self, key, value):
        with Session(self.engine) as session:
            kv = KeyValue(key=key, value=json.dumps(value))
            session.add(kv)
            session.commit()
        return f"Updated and added to value in memory for key: `{key}`"

    def list_keys(self) -> str:
        with Session(self.engine) as session:
            keys = session.query(KeyValue.key).distinct().all()
        content = ["IMPORTANT: your known memcache keys are now:\n"]
        content += [f"- {key[0]}" for key in keys]
        return "\n".join(content)

    def get_all_entries(self) -> str:
        with Session(self.engine) as session:
            entries = session.query(KeyValue).all()
            return [{"key": entry.key, "value": json.loads(entry.value)} for entry in entries]

    def delete_state(self, key: str) -> str:
        with Session(self.engine) as session:
            result = session.query(KeyValue).filter_by(key=key).delete()
            session.commit()
        if result:
            return f"Deleted key `{key}` from memory cache."
        return f"Key `{key}` not found in memory cache."

    def clear_cache(self) -> str:
        with Session(self.engine) as session:
            session.query(KeyValue).delete()
            session.commit()
        return "Cleared all keys in memory cache."

    def snapshot_state(self) -> dict[str, Any]:
        """Return all distinct keys with their merged values."""
        with Session(self.engine) as session:
            keys = [k[0] for k in session.query(KeyValue.key).distinct().all()]
        result = {}
        for k in keys:
            if k.startswith("_log:"):
                result[k] = self.get_log(k.removeprefix("_log:"))
            else:
                result[k] = self.get_state(k)
        return result

    def append_log(self, key: str, entry: Any) -> str:
        from datetime import datetime, timezone

        log_key = f"_log:{key}"
        wrapped = {"_ts": datetime.now(timezone.utc).isoformat(), "data": entry}
        with Session(self.engine) as session:
            kv = KeyValue(key=log_key, value=json.dumps(wrapped))
            session.add(kv)
            session.commit()
        return f"Appended entry to log '{log_key}'"

    def get_log(self, key: str) -> list[dict]:
        """Retrieve all log entries for a key as a flat list."""
        log_key = f"_log:{key}"
        with Session(self.engine) as session:
            rows = session.query(KeyValue).filter_by(key=log_key).order_by(KeyValue.id).all()
        return [json.loads(row.value) for row in rows]
