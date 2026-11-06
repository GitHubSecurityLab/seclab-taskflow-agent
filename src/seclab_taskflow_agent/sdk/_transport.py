# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Transport-level exceptions the backend adapters translate.

Both vendor SDKs run on httpx, and both leak raw transport exceptions
out of streaming responses. The SDK wraps httpx only around the call
that sends the request, and with ``stream=True`` that returns once the
response headers arrive -- the body is iterated afterwards, outside the
handler. A connection that dies mid-body therefore surfaces as a bare
``httpx`` error rather than the SDK's own connection-error type, so the
adapters' ``except <sdk>.APIError`` clauses never see it.

Nothing downstream of an adapter should need to know this. Adapters
translate these into :class:`~.errors.BackendTimeoutError` so the runner
keeps applying retry policy to the neutral hierarchy alone.
"""

from __future__ import annotations

__all__ = ["TRANSIENT_TRANSPORT_ERRORS"]

import httpx

try:  # pragma: no cover - httpx's own dependency, absent only if vendored out
    import httpcore
except ImportError:  # pragma: no cover
    httpcore = None  # type: ignore[assignment]

# Enumerated rather than catching httpx.TransportError wholesale, because
# two of its children are not transient and must not be retried:
# LocalProtocolError means we built a bad request, and UnsupportedProtocol
# is deterministic. Retrying either just repeats the same failure with
# backoff. httpcore's equivalents are included because httpx normally
# remaps them but does not guarantee it on every path.
TRANSIENT_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = tuple(
    exc
    for exc in (
        httpx.TimeoutException,  # read/connect/pool/write timeouts
        httpx.NetworkError,  # connect/read/write/close errors
        httpx.RemoteProtocolError,  # peer closed mid-response
        httpx.ProxyError,
        getattr(httpcore, "TimeoutException", None),
        getattr(httpcore, "NetworkError", None),
        getattr(httpcore, "RemoteProtocolError", None),
        getattr(httpcore, "ProxyError", None),
    )
    if exc is not None
)
