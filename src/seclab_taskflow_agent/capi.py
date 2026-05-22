# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""AI API endpoint and token management.

Supports multiple API providers (GitHub Copilot, GitHub Models, OpenAI, and
custom endpoints).  All provider-specific behaviour is captured in a single
``APIProvider`` dataclass so that adding a new provider only requires one
registry entry instead of changes scattered across multiple match/case blocks.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse

import httpx

__all__ = [
    "COPILOT_INTEGRATION_ID",
    "APIProvider",
    "get_AI_endpoint",
    "get_AI_token",
    "get_provider",
    "list_capi_models",
    "list_tool_call_models",
    "supports_tool_calls",
]

COPILOT_INTEGRATION_ID = os.getenv("COPILOT_INTEGRATION_ID", "vscode-chat")


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class APIProvider:
    """Encapsulates all endpoint-specific behaviour in one place."""

    name: str
    base_url: str
    models_catalog: str = "/models"
    default_model: str = "gpt-4.1"
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure base_url ends with / so httpx URL.join() preserves the path
        if self.base_url and not self.base_url.endswith("/"):
            object.__setattr__(self, "base_url", self.base_url + "/")
        # Freeze mutable headers so singleton providers can't be mutated
        if isinstance(self.extra_headers, dict):
            object.__setattr__(self, "extra_headers", MappingProxyType(self.extra_headers))

    # -- response parsing -----------------------------------------------------

    def parse_models_list(self, body: Any) -> list[dict]:
        """Extract the models list from a catalog response body."""
        if isinstance(body, list):
            return body
        if isinstance(body, dict):
            data = body.get("data", [])
            return data if isinstance(data, list) else []
        return []

    # -- tool-call capability check -------------------------------------------

    def check_tool_calls(self, _model: str, model_info: dict) -> bool:
        """Return True if *model* supports tool calls according to its catalog entry."""
        # Default: optimistically assume support when present in catalog
        return bool(model_info)


class _CopilotProvider(APIProvider):
    """GitHub Copilot API (api.githubcopilot.com)."""

    def check_tool_calls(self, _model: str, model_info: dict) -> bool:
        return (
            model_info
            .get("capabilities", {})
            .get("supports", {})
            .get("tool_calls", False)
        )


class _GitHubModelsProvider(APIProvider):
    """GitHub Models API (models.github.ai)."""

    def parse_models_list(self, body: Any) -> list[dict]:
        # Models API returns a bare list, not {"data": [...]}
        if isinstance(body, list):
            return body
        return super().parse_models_list(body)

    def check_tool_calls(self, _model: str, model_info: dict) -> bool:
        return "tool-calling" in model_info.get("capabilities", [])


class _OpenAIProvider(APIProvider):
    """OpenAI API (api.openai.com).

    The OpenAI /v1/models catalog does not expose capability metadata, so
    we maintain a prefix allowlist of known chat-completion model families.
    """

    _CHAT_PREFIXES = ("gpt-3.5", "gpt-4", "o1", "o3", "o4", "chatgpt-")

    def check_tool_calls(self, _model: str, model_info: dict) -> bool:
        model_id = model_info.get("id", "").lower()
        return any(model_id.startswith(p) for p in self._CHAT_PREFIXES)
# ---------------------------------------------------------------------------
# Provider registry — add new providers here
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, APIProvider] = {
    "api.githubcopilot.com": _CopilotProvider(
        name="copilot",
        base_url="https://api.githubcopilot.com",
        default_model="gpt-4.1",
        extra_headers={"Copilot-Integration-Id": COPILOT_INTEGRATION_ID},
    ),
    "models.github.ai": _GitHubModelsProvider(
        name="github-models",
        base_url="https://models.github.ai/inference",
        models_catalog="/catalog/models",
        default_model="openai/gpt-4.1",
    ),
    "api.openai.com": _OpenAIProvider(
        name="openai",
        base_url="https://api.openai.com/v1",
        models_catalog="/v1/models",
        default_model="gpt-4.1",
    ),
}

def get_provider(endpoint: str | None = None) -> APIProvider:
    """Return the ``APIProvider`` for the given (or configured) endpoint URL.

    When running inside an AWF (Agentic Workflow Firewall) sandbox, the
    ``AWF_COPILOT_PROXY`` env var names the upstream provider whose behaviour
    (headers, model defaults, catalog format) the local proxy mirrors.
    The proxy URL is used as ``base_url`` while all other provider traits
    come from the named upstream.

    ``AWF_COPILOT_PROXY`` accepts either a bare hostname
    (``api.githubcopilot.com``) or a full URL
    (``https://api.githubcopilot.com``).
    """
    url = endpoint or get_AI_endpoint()
    netloc = urlparse(url).netloc
    provider = _PROVIDERS.get(netloc)
    if provider is not None:
        return provider

    # AWF proxy support: AWF_COPILOT_PROXY names the upstream provider
    # (e.g. "api.githubcopilot.com") whose behaviour this proxy mirrors.
    awf_upstream = os.getenv("AWF_COPILOT_PROXY", "").strip()
    if awf_upstream:
        # Normalize: accept both bare hostnames and full URLs.
        parsed = urlparse(awf_upstream)
        key = parsed.netloc or parsed.path
        upstream = _PROVIDERS.get(key)
        if upstream:
            return dataclasses.replace(upstream, base_url=url)

    # Unknown endpoint — return a generic provider with the given base URL
    return APIProvider(name="custom", base_url=url, default_model="please-set-default-model-via-env")


# ---------------------------------------------------------------------------
# Endpoint / token helpers
# ---------------------------------------------------------------------------

def get_AI_endpoint() -> str:
    """Return the configured AI API endpoint URL."""
    return os.getenv("AI_API_ENDPOINT", default="https://models.github.ai/inference")


def get_AI_token() -> str:
    """Get the AI API token from AI_API_TOKEN or COPILOT_TOKEN env vars."""
    token: str | None = os.getenv("AI_API_TOKEN")
    if token:
        return token
    token = os.getenv("COPILOT_TOKEN")
    if token:
        return token
    raise RuntimeError("AI_API_TOKEN environment variable is not set.")


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

def list_capi_models(token: str, endpoint: str | None = None) -> dict[str, dict]:
    """Retrieve available models from the configured API endpoint.

    Args:
        token: Bearer token for authentication.
        endpoint: Optional endpoint URL override (defaults to env config).
    """
    provider = get_provider(endpoint)
    base = provider.base_url
    models: dict[str, dict] = {}
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            **provider.extra_headers,
        }
        r = httpx.get(
            httpx.URL(base).join(provider.models_catalog),
            headers=headers,
        )
        r.raise_for_status()
        for model in provider.parse_models_list(r.json()):
            models[model.get("id")] = dict(model)
    except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError):
        logging.exception("Failed to list models from %s", base)
    return models


def supports_tool_calls(
    model: str,
    models: dict[str, dict],
    endpoint: str | None = None,
) -> bool:
    """Check whether *model* supports tool calls."""
    provider = get_provider(endpoint)
    return provider.check_tool_calls(model, models.get(model, {}))


def list_tool_call_models(token: str, endpoint: str | None = None) -> dict[str, dict]:
    """Return only models that support tool calls."""
    models = list_capi_models(token, endpoint)
    provider = get_provider(endpoint)
    return {
        mid: info
        for mid, info in models.items()
        if provider.check_tool_calls(mid, info)
    }
