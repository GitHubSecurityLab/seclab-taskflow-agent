# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Model resolution logic for Taskflow.

Extracts model configuration and task-level overrides.
"""

from __future__ import annotations

from typing import Any

from .agent import DEFAULT_MODEL
from .available_tools import AvailableTools
from .models import ModelConfigDocument, TaskDefinition


def _resolve_model_config(
    available_tools: AvailableTools,
    model_config_ref: str,
) -> tuple[list[str], dict[str, str], dict[str, dict[str, Any]], str]:
    """Load and validate the model configuration file.

    Args:
        available_tools: Tool registry used to load the config file.
        model_config_ref: Reference name for the model config document.

    Returns:
        A tuple of (model_keys, model_dict, models_params, api_type) where
        model_keys is the list of logical model names, model_dict maps them
        to provider model IDs, models_params holds per-model settings, and
        api_type is ``"chat_completions"`` or ``"responses"``.

    Raises:
        ValueError: If the config file has structural problems.
    """
    m_config: ModelConfigDocument = available_tools.get_model_config(model_config_ref)
    model_dict: dict[str, str] = m_config.models or {}
    model_keys: list[str] = list(model_dict.keys())
    models_params: dict[str, dict[str, Any]] = m_config.model_settings or {}
    unknown = set(models_params) - set(model_keys)
    if unknown:
        raise ValueError(
            f"Settings section of model_config file {model_config_ref} contains models not in the model section: {unknown}"
        )
    return model_keys, model_dict, models_params, m_config.api_type


def _resolve_task_model(
    task: TaskDefinition,
    model_keys: list[str],
    model_dict: dict[str, str],
    models_params: dict[str, dict[str, Any]],
    default_api_type: str = "chat_completions",
) -> tuple[str, dict[str, Any], str, str | None, str | None]:
    """Resolve the final model name, settings, and per-model overrides.

    Returns:
        A tuple of ``(model_id, model_settings, api_type, endpoint, token)``
        where *endpoint* and *token* are ``None`` when not overridden.

    Raises:
        ValueError: If task-level model_settings is not a dictionary.
    """
    logical_name: str = task.model or DEFAULT_MODEL
    model_settings: dict[str, Any] = {}
    api_type: str = default_api_type
    endpoint: str | None = None
    token: str | None = None

    if logical_name in model_keys:
        if logical_name in models_params:
            model_settings = models_params[logical_name].copy()
        logical_name = model_dict[logical_name]

    # Extract engine-level keys before merging task settings
    api_type = model_settings.pop("api_type", api_type)
    endpoint = model_settings.pop("endpoint", None)
    token = model_settings.pop("token", None)

    task_model_settings: dict[str, Any] | Any = task.model_settings or {}
    if not isinstance(task_model_settings, dict):
        raise ValueError(f"model_settings in task {task.name or ''} needs to be a dictionary")

    # Task-level overrides can also set engine keys
    task_settings = dict(task_model_settings)
    api_type = task_settings.pop("api_type", api_type)
    endpoint = task_settings.pop("endpoint", endpoint)
    token = task_settings.pop("token", token)

    model_settings.update(task_settings)
    return logical_name, model_settings, api_type, endpoint, token
