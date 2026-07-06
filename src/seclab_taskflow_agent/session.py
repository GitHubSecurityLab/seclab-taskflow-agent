# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Taskflow session persistence for checkpoint/resume.

Tracks task-level progress through a taskflow so that execution can be
resumed from the last successful checkpoint after an unrecoverable failure.

Session files are stored as JSON in the platformdirs data directory.
"""

from __future__ import annotations

__all__ = [
    "TaskflowSession",
    "artifacts_dir",
    "session_dir",
]

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .path_utils import _data_dir


def session_dir() -> Path:
    """Return (and create) the directory used for session checkpoint files."""
    d = _data_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def artifacts_dir(session_id: str) -> Path:
    """Return (and create) the run-scoped artifacts directory for a session."""
    d = _data_dir() / "artifacts" / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


class TokenUsage(BaseModel):
    """Persisted token usage for one task (or the whole run).

    ``cache_read_tokens`` are input tokens served from the prompt cache (the
    cost saving) and ``cache_write_tokens`` are tokens written to it.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class CompletedTask(BaseModel):
    """Record of a single completed task within a session."""

    index: int
    name: str = ""
    result: bool = False
    skipped: bool = False
    # Resolved model labels this task ran against (one for single-model tasks,
    # several for multi-model). Empty for shell tasks.
    models: list[str] = Field(default_factory=list)
    # Wall-clock duration of the task in seconds.
    duration_s: float = 0.0
    # Token usage attributed to this task (summed across turns and, for
    # multi-model tasks, across every model branch).
    usage: TokenUsage = Field(default_factory=TokenUsage)


class TaskflowSession(BaseModel):
    """Persistent session state for a taskflow run.

    After each task completes the session is saved to disk so that a
    subsequent ``--resume`` invocation can skip already-completed tasks.
    """

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    taskflow_path: str = ""
    cli_globals: dict[str, str] = Field(default_factory=dict)
    prompt: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = ""
    finished_at: str = ""
    completed_tasks: list[CompletedTask] = Field(default_factory=list)
    total_tasks: int = 0
    finished: bool = False
    error: str = ""

    # CLI model config override persisted for deterministic resume
    cli_model_config: str = ""

    # Snapshot of the per-run ResultStore (ordered tool results + named
    # outputs) carried across tasks and restored on resume.
    result_snapshot: dict = Field(default_factory=dict)

    @property
    def next_task_index(self) -> int:
        """Index of the next task to execute."""
        if not self.completed_tasks:
            return 0
        return max(t.index for t in self.completed_tasks) + 1

    @property
    def file_path(self) -> Path:
        """Path to this session's checkpoint file."""
        return session_dir() / f"{self.session_id}.json"

    def save(self) -> Path:
        """Persist session state to disk, returns the file path."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        path = self.file_path
        path.write_text(self.model_dump_json(indent=2))
        logging.debug("Session checkpoint saved: %s", path)
        return path

    def record_task(
        self,
        index: int,
        name: str,
        success: bool,
        result_snapshot: dict | None = None,
        skipped: bool = False,
        models: list[str] | None = None,
        duration_s: float = 0.0,
        usage: dict | None = None,
    ) -> None:
        """Record a completed (or skipped) task and save the checkpoint."""
        self.completed_tasks.append(
            CompletedTask(
                index=index,
                name=name,
                result=success,
                skipped=skipped,
                models=models or [],
                duration_s=duration_s,
                usage=TokenUsage(**(usage or {})),
            )
        )
        if result_snapshot is not None:
            self.result_snapshot = result_snapshot
        self.save()

    def mark_finished(self) -> None:
        """Mark the session as fully completed, write the manifest, and save."""
        self.finished = True
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.save()
        self.write_manifest()

    def mark_failed(self, error: str) -> None:
        """Mark the session as failed, write the manifest, and save."""
        self.error = error
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.save()
        self.write_manifest()

    @property
    def status(self) -> str:
        """Coarse run status: ``finished`` / ``failed`` / ``in_progress``."""
        if self.finished:
            return "finished"
        if self.error:
            return "failed"
        return "in_progress"

    def manifest(self) -> dict:
        """Return a stable, machine-readable summary of the run.

        This is a curated audit view of the checkpoint: per-task status, the
        models each task ran against, timing, token usage, and the named
        outputs (which include per-model fan-in records for multi-model tasks).
        It contains no endpoints or secrets.

        Usage accounting: token usage is reported per recorded task and the
        run-level ``usage`` is the sum over ``completed_tasks``. A non-required
        task that does not complete is still recorded, with ``status="failed"``
        and its usage counted. A task that aborts the run before it can be
        recorded -- an exception, or a ``must_complete`` failure -- is named in
        ``error`` but is not itemized or summed here.
        """
        run_usage = {
            "input_tokens": sum(t.usage.input_tokens for t in self.completed_tasks),
            "output_tokens": sum(t.usage.output_tokens for t in self.completed_tasks),
            "cache_read_tokens": sum(t.usage.cache_read_tokens for t in self.completed_tasks),
            "cache_write_tokens": sum(t.usage.cache_write_tokens for t in self.completed_tasks),
        }
        return {
            "session_id": self.session_id,
            "taskflow": self.taskflow_path,
            "model_config": self.cli_model_config or None,
            "status": self.status,
            "error": self.error or None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at or None,
            "total_tasks": self.total_tasks,
            "usage": run_usage,
            "tasks": [
                {
                    "index": t.index,
                    "name": t.name,
                    "status": "skipped" if t.skipped else ("ok" if t.result else "failed"),
                    "models": t.models,
                    "duration_s": round(t.duration_s, 3),
                    "usage": t.usage.model_dump(),
                }
                for t in self.completed_tasks
            ],
            "outputs": (self.result_snapshot or {}).get("outputs", {}),
        }

    def write_manifest(self) -> Path | None:
        """Write the run manifest to the run-scoped artifacts directory.

        Best-effort: a failure to write the audit artifact must never change
        the run's outcome or mask a task failure, so errors are logged and
        swallowed. Returns the path on success, ``None`` on failure.
        """
        try:
            path = artifacts_dir(self.session_id) / "manifest.json"
            path.write_text(json.dumps(self.manifest(), indent=2, default=str))
            logging.debug("Run manifest written: %s", path)
            return path
        except Exception:  # noqa: BLE001 - audit artifact write is best-effort
            logging.warning("Failed to write run manifest for %s", self.session_id, exc_info=True)
            return None

    @classmethod
    def load(cls, session_id: str) -> TaskflowSession:
        """Load a session from disk by its ID.

        Raises:
            FileNotFoundError: If no checkpoint file exists for the ID.
        """
        path = session_dir() / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No session checkpoint found: {session_id}")
        return cls.model_validate_json(path.read_text())

    @classmethod
    def list_sessions(cls) -> list[TaskflowSession]:
        """List all saved sessions, most recent first."""
        sessions: list[TaskflowSession] = []
        for f in sorted(session_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                sessions.append(cls.model_validate_json(f.read_text()))
            except Exception:
                logging.warning("Skipping corrupt session file: %s", f)
        return sessions
