# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Extended tests for capi module."""

from __future__ import annotations

from seclab_taskflow_agent.capi import get_provider, supports_tool_calls


class TestSupportsToolCalls:
    """Tests for supports_tool_calls with various endpoints."""

    def test_unknown_endpoint_known_model(self, monkeypatch):
        """Unknown endpoint returns True when model is in the catalog."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://custom.api.example.com/v1")
        models = {"my-model": {"id": "my-model"}}
        assert supports_tool_calls("my-model", models) is True

    def test_unknown_endpoint_unknown_model(self, monkeypatch):
        """Unknown endpoint returns False when model is NOT in the catalog."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://custom.api.example.com/v1")
        models = {"other-model": {"id": "other-model"}}
        assert supports_tool_calls("missing-model", models) is False

    def test_copilot_endpoint_with_capabilities(self, monkeypatch):
        """Copilot endpoint checks capabilities.supports.tool_calls."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://api.githubcopilot.com")
        models = {
            "gpt-4o": {
                "id": "gpt-4o",
                "capabilities": {"supports": {"tool_calls": True}},
            }
        }
        assert supports_tool_calls("gpt-4o", models) is True

    def test_copilot_endpoint_without_capabilities(self, monkeypatch):
        """Copilot endpoint returns False when tool_calls not in capabilities."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://api.githubcopilot.com")
        models = {
            "text-only": {
                "id": "text-only",
                "capabilities": {"supports": {}},
            }
        }
        assert supports_tool_calls("text-only", models) is False

    def test_models_github_endpoint(self, monkeypatch):
        """models.github.ai checks for 'tool-calling' in capabilities list."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://models.github.ai/inference")
        models = {
            "openai/gpt-4o": {
                "id": "openai/gpt-4o",
                "capabilities": ["tool-calling", "chat"],
            }
        }
        assert supports_tool_calls("openai/gpt-4o", models) is True

    def test_models_github_endpoint_no_tool_calling(self, monkeypatch):
        """models.github.ai returns False when 'tool-calling' not in list."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://models.github.ai/inference")
        models = {
            "some-model": {
                "id": "some-model",
                "capabilities": ["chat"],
            }
        }
        assert supports_tool_calls("some-model", models) is False

    def test_openai_endpoint_model_in_catalog(self, monkeypatch):
        """OpenAI endpoint returns True for known chat model families."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://api.openai.com/v1")
        models = {"gpt-4o": {"id": "gpt-4o"}}
        assert supports_tool_calls("gpt-4o", models) is True

    def test_openai_endpoint_o_series(self, monkeypatch):
        """OpenAI endpoint returns True for o-series reasoning models."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://api.openai.com/v1")
        for mid in ("o1-preview", "o3-mini", "o4-mini"):
            models = {mid: {"id": mid}}
            assert supports_tool_calls(mid, models) is True

    def test_openai_endpoint_non_chat_model(self, monkeypatch):
        """OpenAI endpoint returns False for embeddings/audio/image models."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://api.openai.com/v1")
        for mid in ("text-embedding-ada-002", "whisper-1", "dall-e-3", "tts-1"):
            models = {mid: {"id": mid}}
            assert supports_tool_calls(mid, models) is False

    def test_openai_endpoint_model_not_in_catalog(self, monkeypatch):
        """OpenAI endpoint returns False when model is not in catalog."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://api.openai.com/v1")
        assert supports_tool_calls("missing-model", {}) is False

    def test_explicit_endpoint_override(self):
        """supports_tool_calls accepts an explicit endpoint parameter."""
        models = {"my-model": {"id": "my-model", "capabilities": {"supports": {"tool_calls": True}}}}
        assert supports_tool_calls("my-model", models, endpoint="https://api.githubcopilot.com") is True


class TestGetProvider:
    """Tests for the provider registry."""

    def test_copilot_provider(self):
        p = get_provider("https://api.githubcopilot.com")
        assert p.name == "copilot"
        assert p.base_url == "https://api.githubcopilot.com/"
        assert "Copilot-Integration-Id" in p.extra_headers

    def test_github_models_provider(self):
        p = get_provider("https://models.github.ai/inference")
        assert p.name == "github-models"
        assert p.models_catalog == "/catalog/models"
        assert p.default_model == "openai/gpt-4.1"

    def test_openai_provider(self):
        p = get_provider("https://api.openai.com/v1")
        assert p.name == "openai"
        assert not p.extra_headers

    def test_custom_endpoint(self):
        p = get_provider("https://my-custom-llm.example.com/v1")
        assert p.name == "custom"
        assert p.base_url == "https://my-custom-llm.example.com/v1/"
        assert not p.extra_headers

    def test_awf_proxy_bare_hostname(self, monkeypatch):
        monkeypatch.setenv("AWF_COPILOT_PROXY", "api.githubcopilot.com")
        p = get_provider("http://172.30.0.30:10002")
        assert p.name == "copilot"
        assert p.base_url == "http://172.30.0.30:10002/"
        assert p.default_model == "gpt-4.1"
        assert "Copilot-Integration-Id" in p.extra_headers

    def test_awf_proxy_full_url(self, monkeypatch):
        monkeypatch.setenv("AWF_COPILOT_PROXY", "https://api.githubcopilot.com")
        p = get_provider("http://172.30.0.30:10002")
        assert p.name == "copilot"
        assert p.base_url == "http://172.30.0.30:10002/"

    def test_awf_proxy_unknown_upstream(self, monkeypatch):
        monkeypatch.setenv("AWF_COPILOT_PROXY", "not-a-real-provider.com")
        p = get_provider("http://172.30.0.30:10002")
        assert p.name == "custom"

    def test_awf_proxy_not_set(self):
        p = get_provider("http://172.30.0.30:10002")
        assert p.name == "custom"
