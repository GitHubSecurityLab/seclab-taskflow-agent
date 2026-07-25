# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Neutral backend error hierarchy.

Adapters translate their SDK-native exceptions into these so the runner
can apply retry/backoff policy without importing any SDK.
"""

from __future__ import annotations

__all__ = [
    "BackendBadRequestError",
    "BackendCapabilityError",
    "BackendError",
    "BackendMaxTurnsError",
    "BackendQuotaExhaustedError",
    "BackendRateLimitError",
    "BackendTimeoutError",
    "BackendUnexpectedError",
]


class BackendError(Exception):
    """Base class for all neutral backend errors."""


class BackendCapabilityError(BackendError):
    """The active backend does not support a requested YAML feature.

    Raised at task build time so misconfiguration fails before any
    network call is made. The message names the offending field and the
    backend that rejected it.
    """


class BackendTimeoutError(BackendError):
    """The backend timed out waiting on the upstream API."""


class BackendRateLimitError(BackendError):
    """The backend was rate-limited by the upstream API.

    This error is retryable with appropriate backoff.
    """


class BackendQuotaExhaustedError(BackendError):
    """The backend quota has been exhausted (e.g., token budget, request limit).

    Unlike rate limiting, quota exhaustion is not retryable within the current
    billing period or allocation cycle. The caller should check their usage
    limits or wait until the quota resets.
    """


class BackendBadRequestError(BackendError):
    """The backend rejected the request (typically 4xx)."""


class BackendMaxTurnsError(BackendError):
    """The backend exceeded the configured maximum agent turn count."""


class BackendUnexpectedError(BackendError):
    """Any other backend exception that does not fit a specific class."""
