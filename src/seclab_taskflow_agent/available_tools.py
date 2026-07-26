# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""YAML resource loader for taskflow grammar files.

Loads and caches personality, taskflow, toolbox, model_config, and prompt
YAML files, validating them against Pydantic grammar models at parse time.
"""

from __future__ import annotations

__all__ = ["AvailableTools"]

import importlib.resources
from enum import Enum
from typing import Union

import yaml
from pydantic import ValidationError

from .models import (
    DOCUMENT_MODELS,
    ModelConfigDocument,
    PersonalityDocument,
    PromptDocument,
    TaskflowDocument,
    ToolboxDocument,
)


class BadToolNameError(Exception):
    pass


class VersionException(Exception):
    pass


class FileTypeException(Exception):
    pass


class AvailableToolType(Enum):
    Personality = "personality"
    Taskflow = "taskflow"
    Prompt = "prompt"
    Toolbox = "toolbox"
    ModelConfig = "model_config"


# Union of all document model types returned by AvailableTools
DocumentModel = Union[
    TaskflowDocument, PersonalityDocument, ToolboxDocument,
    ModelConfigDocument, PromptDocument,
]


class AvailableTools:
    """Loads, validates, and caches YAML grammar files as Pydantic models."""

    def __init__(self) -> None:
        self._cache: dict[AvailableToolType, dict[str, DocumentModel]] = {}

    def get_personality(self, name: str) -> PersonalityDocument:
        """Load a personality YAML and return a validated PersonalityDocument."""
        return self._load(AvailableToolType.Personality, name)

    def get_taskflow(self, name: str) -> TaskflowDocument:
        """Load a taskflow YAML and return a validated TaskflowDocument."""
        return self._load(AvailableToolType.Taskflow, name)

    def get_prompt(self, name: str) -> PromptDocument:
        """Load a prompt YAML and return a validated PromptDocument."""
        return self._load(AvailableToolType.Prompt, name)

    def get_toolbox(self, name: str) -> ToolboxDocument:
        """Load a toolbox YAML and return a validated ToolboxDocument."""
        return self._load(AvailableToolType.Toolbox, name)

    def get_model_config(self, name: str) -> ModelConfigDocument:
        """Load a model_config YAML and return a validated ModelConfigDocument."""
        return self._load(AvailableToolType.ModelConfig, name)

    # Keep legacy alias for code that uses the generic accessor
    def get_tool(self, tooltype: AvailableToolType, toolname: str) -> DocumentModel:
        """Generic loader — prefer the typed ``get_*()`` methods."""
        return self._load(tooltype, toolname)

    def _load(self, tooltype: AvailableToolType, toolname: str) -> DocumentModel:
        """Load, validate, and cache a YAML grammar file.

        Args:
            tooltype: Expected file type (personality, taskflow, etc.).
            toolname: Dotted module path, e.g. ``"examples.taskflows.echo"``.

        Returns:
            A validated Pydantic document model instance.

        Raises:
            BadToolNameError: If the tool cannot be found or loaded.
            VersionException: If the grammar version is unsupported.
            FileTypeException: If the filetype doesn't match expectations.
        """
        # Check cache first
        if tooltype in self._cache and toolname in self._cache[tooltype]:
            return self._cache[tooltype][toolname]

        # Resolve package and filename from dotted path
        components = toolname.rsplit(".", 1)
        if len(components) != 2:
            raise BadToolNameError(
                f'Invalid tool name format: "{toolname}".\n'
                f'Expected format: "package.module" (e.g., "examples.taskflows.echo").\n'
                f'Please provide a dotted path to a YAML file without the .yaml extension.'
            )
        package, filename = components

        try:
            pkg_dir = importlib.resources.files(package)
            if not pkg_dir.is_dir():
                raise BadToolNameError(
                    f"Cannot find package '{package}' for tool '{toolname}'.\n"
                    f"The package directory does not exist or is not accessible.\n"
                    f"Please verify the package name is correct and the package is installed."
                )
            filepath = pkg_dir.joinpath(filename + ".yaml")
            with filepath.open(encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)

            # Validate header before full parse
            header = raw.get("seclab-taskflow-agent", {})
            filetype = header.get("filetype", "")
            if filetype != tooltype.value:
                raise FileTypeException(
                    f"File type mismatch in {filepath}.\n"
                    f"Expected filetype: '{tooltype.value}'\n"
                    f"Actual filetype: '{filetype}'\n"
                    f"Please ensure the file is the correct type ({tooltype.value})."
                )

            # Parse into the appropriate Pydantic model
            model_cls = DOCUMENT_MODELS.get(filetype)
            if model_cls is None:
                raise BadToolNameError(
                    f"Unknown file type '{filetype}' in {toolname}.\n"
                    f"Supported file types: {', '.join(DOCUMENT_MODELS.keys())}.\n"
                    f"Please check the 'filetype' field in the YAML header."
                )

            try:
                doc = model_cls(**raw)
            except ValidationError as exc:
                # Surface version errors as VersionException for compat
                for err in exc.errors():
                    if "Unsupported version" in str(err.get("msg", "")):
                        raise VersionException(str(err["msg"])) from exc
                # Format validation errors in a more user-friendly way
                error_details = []
                for err in exc.errors():
                    loc = " -> ".join(str(x) for x in err.get("loc", []))
                    msg = err.get("msg", "")
                    error_details.append(f"  - {loc}: {msg}")
                formatted_errors = "\n".join(error_details)
                raise BadToolNameError(
                    f"Validation error loading {toolname}:\n"
                    f"{formatted_errors}\n"
                    f"Please check the YAML file for missing or invalid fields."
                ) from exc

            # Cache and return
            if tooltype not in self._cache:
                self._cache[tooltype] = {}
            self._cache[tooltype][toolname] = doc
            return doc

        except ModuleNotFoundError as exc:
            raise BadToolNameError(
                f"Cannot find module '{package}' for tool '{toolname}'.\n"
                f"Original error: {exc}\n"
                f"Please verify:\n"
                f"  1. The package '{package}' is installed\n"
                f"  2. The package is in your Python path\n"
                f"  3. The tool name format is correct (package.module)"
            ) from exc
        except FileNotFoundError:
            raise BadToolNameError(
                f"Cannot find file for tool '{toolname}'.\n"
                f"Expected file: {filepath}\n"
                f"The file does not exist. Please verify:\n"
                f"  1. The file path is correct\n"
                f"  2. The file exists in the specified location\n"
                f"  3. The file name does not include the .yaml extension in the tool name"
            )
        except yaml.YAMLError as exc:
            # Provide helpful YAML syntax error messages
            marker = getattr(exc, "problem_mark", None)
            if marker:
                line_no = marker.line + 1
                col_no = marker.column + 1
                raise BadToolNameError(
                    f"YAML syntax error in {toolname} at line {line_no}, column {col_no}.\n"
                    f"Error: {exc.problem}\n"
                    f"Please check the YAML syntax around this location."
                ) from exc
            else:
                raise BadToolNameError(
                    f"YAML syntax error in {toolname}.\n"
                    f"Error: {exc}\n"
                    f"Please check the YAML file for syntax errors."
                ) from exc
        except ValueError as exc:
            raise BadToolNameError(f"Cannot load {toolname}: {exc}") from exc
