# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

# CAPI specific interactions
import json
import logging
import os
from urllib.parse import urlparse

import httpx
from strenum import StrEnum


# Enumeration of currently supported API endpoints.
class AI_API_ENDPOINT_ENUM(StrEnum):
    AI_API_MODELS_GITHUB = "models.github.ai"
    AI_API_GITHUBCOPILOT = "api.githubcopilot.com"
    AI_API_OPENAI = "api.openai.com"

    def to_url(self):
        """
        Convert the endpoint to its full URL.
        """
        match self:
            case AI_API_ENDPOINT_ENUM.AI_API_GITHUBCOPILOT:
                return f"https://{self}"
            case AI_API_ENDPOINT_ENUM.AI_API_MODELS_GITHUB:
                return f"https://{self}/inference"
            case AI_API_ENDPOINT_ENUM.AI_API_OPENAI:
                return f"https://{self}/v1"
            case _:
                raise ValueError(f"Unsupported endpoint: {self}")


COPILOT_INTEGRATION_ID = "vscode-chat"


# you can also set https://api.githubcopilot.com if you prefer
# but beware that your taskflows need to reference the correct model id
# since different APIs use their own id schema, use -l with your desired
# endpoint to retrieve the correct id names to use for your taskflow
def get_AI_endpoint():
    return os.getenv("AI_API_ENDPOINT", default="https://models.github.ai/inference")


def get_AI_token():
    """
    Get the token for the AI API from the environment.
    The environment variable can be named either AI_API_TOKEN,
    COPILOT_TOKEN, or GITHUB_TOKEN.
    """
    token = os.getenv("AI_API_TOKEN")
    if token:
        return token
    token = os.getenv("COPILOT_TOKEN")
    if token:
        return token
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return token
    raise RuntimeError("AI_API_TOKEN environment variable is not set.")


# assume we are >= python 3.9 for our type hints
def list_capi_models(token: str) -> dict[str, dict]:
    """Retrieve a dictionary of available CAPI models"""
    models = {}
    try:
        api_endpoint = get_AI_endpoint()
        netloc = urlparse(api_endpoint).netloc
        match netloc:
            case AI_API_ENDPOINT_ENUM.AI_API_GITHUBCOPILOT:
                models_catalog = "models"
            case AI_API_ENDPOINT_ENUM.AI_API_MODELS_GITHUB:
                models_catalog = "catalog/models"
            case AI_API_ENDPOINT_ENUM.AI_API_OPENAI:
                models_catalog = "models"
            case _:
                raise ValueError(
                    f"Unsupported Model Endpoint: {api_endpoint}\n"
                    f"Supported endpoints: {[e.to_url() for e in AI_API_ENDPOINT_ENUM]}"
                )
        r = httpx.get(
            httpx.URL(api_endpoint).join(models_catalog),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
                "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
            },
        )
        r.raise_for_status()
        # CAPI vs Models API
        match netloc:
            case AI_API_ENDPOINT_ENUM.AI_API_GITHUBCOPILOT:
                models_list = r.json().get("data", [])
            case AI_API_ENDPOINT_ENUM.AI_API_MODELS_GITHUB:
                models_list = r.json()
            case AI_API_ENDPOINT_ENUM.AI_API_OPENAI:
                models_list = r.json().get("data", [])
        for model in models_list:
            models[model.get("id")] = dict(model)
    except httpx.RequestError as e:
        logging.exception("Request error")
    except json.JSONDecodeError as e:
        logging.exception("JSON error")
    except httpx.HTTPStatusError as e:
        logging.exception("HTTP error")
    return models


def supports_tool_calls(model: str, models: dict) -> bool:
    api_endpoint = get_AI_endpoint()
    match urlparse(api_endpoint).netloc:
        case AI_API_ENDPOINT_ENUM.AI_API_GITHUBCOPILOT:
            return models.get(model, {}).get("capabilities", {}).get("supports", {}).get("tool_calls", False)
        case AI_API_ENDPOINT_ENUM.AI_API_MODELS_GITHUB:
            return "tool-calling" in models.get(model, {}).get("capabilities", [])
        case AI_API_ENDPOINT_ENUM.AI_API_OPENAI:
            # OpenAI doesn't expose capabilities in the models list
            # Check if model name indicates function calling support
            model_lower = model.lower()
            return any(
                [
                    "gpt-" in model_lower,
                ]
            )
        case _:
            raise ValueError(
                f"Unsupported Model Endpoint: {api_endpoint}\n"
                f"Supported endpoints: {[e.to_url() for e in AI_API_ENDPOINT_ENUM]}"
            )


def list_tool_call_models(token: str) -> dict[str, dict]:
    models = list_capi_models(token)
    tool_models = {}
    for model in models:
        if supports_tool_calls(model, models) is True:
            tool_models[model] = models[model]
    return tool_models
