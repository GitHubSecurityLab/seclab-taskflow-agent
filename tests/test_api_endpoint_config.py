# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""
Test API endpoint configuration.
"""

import os
from urllib.parse import urlparse

import pytest

from seclab_taskflow_agent.capi import get_AI_endpoint, get_provider, list_capi_models


class TestAPIEndpoint:
    """Test API endpoint configuration."""

    def test_default_api_endpoint(self):
        """Test that default API endpoint is set to models.github.ai/inference."""
        try:
            original_env = os.environ.pop("AI_API_ENDPOINT", None)
            endpoint = get_AI_endpoint()
            assert endpoint is not None
            assert isinstance(endpoint, str)
            assert urlparse(endpoint).netloc == "api.githubcopilot.com"
        finally:
            if original_env:
                os.environ["AI_API_ENDPOINT"] = original_env

    def test_api_endpoint_env_override(self):
        """Test that AI_API_ENDPOINT can be overridden by environment variable."""
        try:
            original_env = os.environ.pop("AI_API_ENDPOINT", None)
            test_endpoint = "https://api.githubcopilot.com"
            os.environ["AI_API_ENDPOINT"] = test_endpoint
            assert get_AI_endpoint() == test_endpoint
        finally:
            if original_env:
                os.environ["AI_API_ENDPOINT"] = original_env

    def test_provider_base_urls(self):
        """Test that providers resolve to expected base URLs."""
        assert get_provider("https://models.github.ai/inference").base_url == "https://models.github.ai/inference/"
        assert get_provider("https://api.githubcopilot.com").base_url == "https://api.githubcopilot.com/"
        assert get_provider("https://api.openai.com/v1").base_url == "https://api.openai.com/v1/"

    def test_unsupported_endpoint(self, monkeypatch):
        """Test that unsupported API endpoint falls back gracefully."""
        api_endpoint = "https://unsupported.example.com"
        monkeypatch.setenv("AI_API_ENDPOINT", api_endpoint)
        result = list_capi_models("abc")
        assert isinstance(result, dict)
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
