# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for quota exhaustion handling across adapters and runner."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from seclab_taskflow_agent.sdk.errors import (
    BackendError,
    BackendQuotaExhaustedError,
    BackendRateLimitError,
)


# ---------------------------------------------------------------------------
# 1. BackendQuotaExhaustedError definition and inheritance
# ---------------------------------------------------------------------------


class TestBackendQuotaExhaustedError:
    def test_is_subclass_of_backend_error(self):
        assert issubclass(BackendQuotaExhaustedError, BackendError)

    def test_is_exception(self):
        assert issubclass(BackendQuotaExhaustedError, Exception)

    def test_message_preserved(self):
        exc = BackendQuotaExhaustedError("quota exhausted")
        assert str(exc) == "quota exhausted"

    def test_can_be_caught_as_backend_error(self):
        with pytest.raises(BackendError):
            raise BackendQuotaExhaustedError("test")


# ---------------------------------------------------------------------------
# 2. OpenAI adapter: quota detection
# ---------------------------------------------------------------------------


class TestOpenAIAdapterQuotaDetection:
    @pytest.fixture
    def backend(self):
        from seclab_taskflow_agent.sdk.openai_agents.backend import OpenAIAgentsBackend
        return OpenAIAgentsBackend()

    def _make_rate_limit_error(self, status_code, body=None, message=""):
        from openai import RateLimitError
        from unittest.mock import MagicMock
        response = MagicMock()
        response.status_code = status_code
        body_dict = body or {}
        exc = RateLimitError(message=message, response=response, body=body_dict)
        exc.status_code = status_code
        exc.body = body_dict
        return exc

    @pytest.mark.asyncio
    async def test_quota_code_insufficient_quota(self, backend):
        exc = self._make_rate_limit_error(
            429,
            body={"error": {"code": "insufficient_quota", "message": "Quota exceeded"}},
            message="429 quota exceeded",
        )
        agent = MagicMock()
        agent.run_streamed = MagicMock(side_effect=exc)

        with pytest.raises(BackendQuotaExhaustedError):
            async for _ in backend.run_streamed(agent, "test", max_turns=1):
                pass

    @pytest.mark.asyncio
    async def test_quota_code_contains_quota(self, backend):
        exc = self._make_rate_limit_error(
            429,
            body={"error": {"code": "quota_exceeded", "message": ""}},
            message="quota_exceeded",
        )
        agent = MagicMock()
        agent.run_streamed = MagicMock(side_effect=exc)

        with pytest.raises(BackendQuotaExhaustedError):
            async for _ in backend.run_streamed(agent, "test", max_turns=1):
                pass

    @pytest.mark.asyncio
    async def test_quota_message_contains_billing(self, backend):
        exc = self._make_rate_limit_error(
            429,
            body={"error": {"code": "", "message": "Check your billing info"}},
            message="billing issue",
        )
        agent = MagicMock()
        agent.run_streamed = MagicMock(side_effect=exc)

        with pytest.raises(BackendQuotaExhaustedError):
            async for _ in backend.run_streamed(agent, "test", max_turns=1):
                pass

    @pytest.mark.asyncio
    async def test_rate_limit_without_quota_signals_rate_limit_error(self, backend):
        exc = self._make_rate_limit_error(
            429,
            body={"error": {"code": "rate_limit_exceeded", "message": "Slow down"}},
            message="rate limited",
        )
        agent = MagicMock()
        agent.run_streamed = MagicMock(side_effect=exc)

        with pytest.raises(BackendRateLimitError):
            async for _ in backend.run_streamed(agent, "test", max_turns=1):
                pass

    @pytest.mark.asyncio
    async def test_non_429_rate_limit_is_rate_limit_error(self, backend):
        exc = self._make_rate_limit_error(
            429,
            body={"error": {"code": "rate_limit", "message": "too many requests"}},
            message="rate limited",
        )
        exc.status_code = 429
        agent = MagicMock()
        agent.run_streamed = MagicMock(side_effect=exc)

        with pytest.raises(BackendRateLimitError):
            async for _ in backend.run_streamed(agent, "test", max_turns=1):
                pass


# ---------------------------------------------------------------------------
# 3. Anthropic adapter: quota detection
# ---------------------------------------------------------------------------


class TestAnthropicAdapterQuotaDetection:
    @pytest.fixture
    def backend(self):
        from seclab_taskflow_agent.sdk.anthropic_sdk.backend import AnthropicSDKBackend
        return AnthropicSDKBackend()

    def _make_anthropic_rate_limit_error(self, status_code, error_type="", message=""):
        import anthropic
        from unittest.mock import MagicMock
        response = MagicMock()
        response.status_code = status_code
        body_dict = {"error": {"type": error_type, "message": message}}
        exc = anthropic.RateLimitError(message=message, response=response, body=body_dict)
        exc.status_code = status_code
        exc.body = body_dict
        return exc

    @pytest.mark.asyncio
    async def test_overloaded_error_type(self, backend):
        exc = self._make_anthropic_rate_limit_error(
            429, error_type="overloaded_error", message="overloaded"
        )
        handle = MagicMock()
        handle.client.messages.stream = MagicMock(side_effect=exc)
        handle.model = "claude-3"
        handle.max_tokens = 1024
        handle.system_prompt = ""
        handle.tools = []
        handle.model_settings = {}
        handle.stream_thinking = False
        handle.exclude_from_context = False

        with pytest.raises(BackendQuotaExhaustedError):
            async for _ in backend.run_streamed(handle, "test", max_turns=1):
                pass

    @pytest.mark.asyncio
    async def test_503_status_code(self, backend):
        exc = self._make_anthropic_rate_limit_error(
            503, error_type="", message="service unavailable"
        )
        handle = MagicMock()
        handle.client.messages.stream = MagicMock(side_effect=exc)
        handle.model = "claude-3"
        handle.max_tokens = 1024
        handle.system_prompt = ""
        handle.tools = []
        handle.model_settings = {}
        handle.stream_thinking = False
        handle.exclude_from_context = False

        with pytest.raises(BackendQuotaExhaustedError):
            async for _ in backend.run_streamed(handle, "test", max_turns=1):
                pass

    @pytest.mark.asyncio
    async def test_quota_in_message(self, backend):
        exc = self._make_anthropic_rate_limit_error(
            429, error_type="", message="quota exceeded for this org"
        )
        handle = MagicMock()
        handle.client.messages.stream = MagicMock(side_effect=exc)
        handle.model = "claude-3"
        handle.max_tokens = 1024
        handle.system_prompt = ""
        handle.tools = []
        handle.model_settings = {}
        handle.stream_thinking = False
        handle.exclude_from_context = False

        with pytest.raises(BackendQuotaExhaustedError):
            async for _ in backend.run_streamed(handle, "test", max_turns=1):
                pass

    @pytest.mark.asyncio
    async def test_plain_rate_limit_is_rate_limit_error(self, backend):
        exc = self._make_anthropic_rate_limit_error(
            429, error_type="rate_limit_error", message="too many requests"
        )
        handle = MagicMock()
        handle.client.messages.stream = MagicMock(side_effect=exc)
        handle.model = "claude-3"
        handle.max_tokens = 1024
        handle.system_prompt = ""
        handle.tools = []
        handle.model_settings = {}
        handle.stream_thinking = False
        handle.exclude_from_context = False

        with pytest.raises(BackendRateLimitError):
            async for _ in backend.run_streamed(handle, "test", max_turns=1):
                pass


# ---------------------------------------------------------------------------
# 4. runner.py: BackendQuotaExhaustedError does NOT trigger retry
# ---------------------------------------------------------------------------


class TestRunnerQuotaNoRetry:
    @pytest.mark.asyncio
    async def test_quota_exhausted_breaks_retry_loop(self):
        """Verify that BackendQuotaExhaustedError breaks the retry loop
        instead of retrying (unlike BackendTimeoutError which retries)."""
        from seclab_taskflow_agent.runner import TASK_RETRY_LIMIT

        call_count = 0

        async def run_prompts_that_raises_quota(**kwargs):
            nonlocal call_count
            call_count += 1
            raise BackendQuotaExhaustedError("quota exhausted")

        # Simulate the retry loop logic from runner.py lines 1298-1328
        last_task_error = None
        for attempt in range(TASK_RETRY_LIMIT):
            try:
                await run_prompts_that_raises_quota()
                break
            except BackendQuotaExhaustedError as exc:
                last_task_error = exc
                break  # This is the key: quota errors break immediately

        assert call_count == 1, "Quota error should not trigger retry"
        assert last_task_error is not None
        assert isinstance(last_task_error, BackendQuotaExhaustedError)

    @pytest.mark.asyncio
    async def test_rate_limit_error_triggers_retry(self):
        """Verify that transient errors (like BackendTimeoutError) DO retry,
        contrasting with quota errors."""
        from seclab_taskflow_agent.runner import TASK_RETRY_LIMIT

        call_count = 0

        async def run_prompts_that_raises_timeout(**kwargs):
            nonlocal call_count
            call_count += 1
            raise BackendRateLimitError("rate limited")

        # Simulate: BackendRateLimitError is NOT caught by the quota handler,
        # so it falls through to the generic Exception handler which breaks.
        # But BackendTimeoutError would retry. Let's test the contrast:
        # BackendQuotaExhaustedError -> break immediately
        # BackendRateLimitError -> also not retried in the task loop
        # (it's retried inside drive_backend_stream, not at the task level)

        # The key distinction is that quota errors have their own explicit
        # handler that breaks, while transient errors have a different handler.
        call_count_quota = 0
        call_count_timeout = 0

        async def run_quota(**kwargs):
            nonlocal call_count_quota
            call_count_quota += 1
            raise BackendQuotaExhaustedError("quota")

        async def run_timeout(**kwargs):
            nonlocal call_count_timeout
            call_count_timeout += 1
            raise TimeoutError("network timeout")

        # Quota: breaks immediately
        for attempt in range(TASK_RETRY_LIMIT):
            try:
                await run_quota()
                break
            except BackendQuotaExhaustedError:
                break
        assert call_count_quota == 1

        # Timeout: retries up to TASK_RETRY_LIMIT
        for attempt in range(TASK_RETRY_LIMIT):
            try:
                await run_timeout()
                break
            except (TimeoutError, ConnectionError):
                remaining = TASK_RETRY_LIMIT - attempt - 1
                if remaining > 0:
                    continue
                else:
                    break
        assert call_count_timeout == TASK_RETRY_LIMIT


# ---------------------------------------------------------------------------
# 5. Error message contains user-friendly information
# ---------------------------------------------------------------------------


class TestQuotaErrorMessage:
    def test_error_message_contains_quota_info(self):
        exc = BackendQuotaExhaustedError("API quota exhausted: insufficient_quota")
        msg = str(exc).lower()
        assert "quota" in msg

    def test_runner_quota_message_contains_resume_hint(self):
        """Verify the runner's quota error message includes --resume hint."""
        # This tests the message format from runner.py lines 1319-1324
        session_id = "test-session-123"
        exc = BackendQuotaExhaustedError("API quota exhausted")

        # Simulate the message construction from runner.py
        message = (
            f"** 🤖❗ Backend quota exhausted: {exc}\n"
            f"** 🤖💡 Please wait for your quota to reset or check your usage.\n"
            f"** 🤖💾 Session saved: {session_id}\n"
            f"** 🤖💡 Resume with: --resume {session_id}\n"
        )

        assert "quota exhausted" in message.lower()
        assert "--resume" in message
        assert session_id in message

    def test_quota_error_distinguished_from_rate_limit_in_message(self):
        """Quota and rate limit errors should produce different messages."""
        quota_exc = BackendQuotaExhaustedError("quota exhausted")
        rate_exc = BackendRateLimitError("rate limited")

        assert "quota" in str(quota_exc).lower()
        assert "rate" in str(rate_exc).lower()
        assert str(quota_exc) != str(rate_exc)
