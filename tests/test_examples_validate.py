# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Corpus gate: every bundled/example grammar file must validate and every
example taskflow must lint without errors.

This turns an accidental breakage of a shipped example (or a grammar change
that invalidates one) into an instant, local test failure instead of a
runtime surprise.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import yaml

from seclab_taskflow_agent.available_tools import AvailableTools
from seclab_taskflow_agent.linting import lint_taskflow
from seclab_taskflow_agent.models import DOCUMENT_MODELS

# Roots that contain shipped grammar documents.
_ROOTS = ["examples", "src/seclab_taskflow_agent"]


def _grammar_files() -> list[str]:
    files: list[str] = []
    for root in _ROOTS:
        files.extend(sorted(glob.glob(f"{root}/**/*.yaml", recursive=True)))
    # Keep only files with a taskflow-agent header.
    kept: list[str] = []
    for f in files:
        try:
            data = yaml.safe_load(Path(f).read_text(encoding='utf-8'))
        except Exception:  # noqa: BLE001 - reported by the validation test
            kept.append(f)
            continue
        if isinstance(data, dict) and "seclab-taskflow-agent" in data:
            kept.append(f)
    return kept


def _dotted(path: str) -> str:
    p = path.removeprefix("src/")
    return p[: -len(".yaml")].replace("/", ".")


def _taskflow_dotted_paths() -> list[str]:
    paths: list[str] = []
    for f in sorted(glob.glob("examples/taskflows/**/*.yaml", recursive=True)):
        data = yaml.safe_load(Path(f).read_text(encoding='utf-8'))
        if isinstance(data, dict) and (data.get("seclab-taskflow-agent") or {}).get("filetype") == "taskflow":
            paths.append(_dotted(f))
    return paths


@pytest.mark.parametrize("path", _grammar_files())
def test_bundled_document_validates(path: str) -> None:
    """Every shipped grammar document parses and validates against its model."""
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    assert isinstance(data, dict), f"{path}: not a mapping"
    filetype = (data.get("seclab-taskflow-agent") or {}).get("filetype")
    model = DOCUMENT_MODELS.get(filetype)
    assert model is not None, f"{path}: unknown filetype {filetype!r}"
    # Raises ValidationError (failing the test) if the document is invalid.
    model.model_validate(data)


@pytest.mark.parametrize("dotted", _taskflow_dotted_paths())
def test_example_taskflow_lints_without_errors(dotted: str) -> None:
    """Every example taskflow lints clean (warnings allowed, no errors)."""
    at = AvailableTools()
    issues = lint_taskflow(at, dotted)
    errors = [i for i in issues if i.severity == "error"]
    assert not errors, f"{dotted} has lint errors:\n" + "\n".join(
        f"  {i.code}: {i.message} [{i.location}]" for i in errors
    )
