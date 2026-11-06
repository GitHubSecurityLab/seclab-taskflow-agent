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
    "BackendRateLimitError",
    "BackendTimeoutError",
    "BackendTransportError",
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


class BackendTransportError(BackendError):
    """The connection failed partway through a response.

    Deliberately a sibling of :class:`BackendTimeoutError` rather than a
    subclass. ``drive_backend_stream`` retries ``BackendTimeoutError``
    in-place, which is right for an idle stream -- nothing was emitted, so
    replaying costs only the turn. It is wrong here: by the time a stream
    dies mid-body the agent has already run tool calls and their results
    have already been recorded, and that loop does not reset capture state,
    so each in-place retry appends another copy.

    Retry for this class belongs to ``_deploy_with_transport_retry``, which
    resets the branch's capture state between attempts and leaves completed
    siblings alone. Keeping the classes distinct is what stops the two loops
    from multiplying.
    """


class BackendRateLimitError(BackendError):
    """The backend was rate-limited by the upstream API."""


class BackendBadRequestError(BackendError):
    """The backend rejected the request (typically 4xx)."""


class BackendMaxTurnsError(BackendError):
    """The backend exceeded the configured maximum agent turn count."""


class BackendUnexpectedError(BackendError):
    """Any other backend exception that does not fit a specific class."""
