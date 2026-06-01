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
    "session_dir",
]

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


class CompletedTask(BaseModel):
    """Record of a single completed task within a session."""

    index: int
    name: str = ""
    result: bool = False
    tool_results: list[str] = Field(default_factory=list)


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
    completed_tasks: list[CompletedTask] = Field(default_factory=list)
    total_tasks: int = 0
    finished: bool = False
    error: str = ""

    # Accumulated tool results carried across tasks (used by repeat_prompt)
    last_tool_results: list[str] = Field(default_factory=list)

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
        tool_results: list[str] | None = None,
    ) -> None:
        """Record a completed task and save the checkpoint."""
        self.completed_tasks.append(
            CompletedTask(
                index=index,
                name=name,
                result=success,
                tool_results=tool_results or [],
            )
        )
        self.last_tool_results = list(tool_results or [])
        self.save()

    def mark_finished(self) -> None:
        """Mark the session as fully completed and save."""
        self.finished = True
        self.save()

    def mark_failed(self, error: str) -> None:
        """Mark the session as failed with an error message and save."""
        self.error = error
        self.save()

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
