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
translate these into :class:`~.errors.BackendTransportError` so the
runner keeps applying retry policy to the neutral hierarchy alone.

Every installed httpx family is enumerated, not just ``httpx``. Some
environments ship both ``httpx``/``httpcore`` and ``httpx2``/``httpcore2``,
and their exception classes are unrelated: ``httpx.RemoteProtocolError is
httpx2.RemoteProtocolError`` is False. A long-running task died on
``httpx2.RemoteProtocolError`` while this tuple listed only the ``httpx``
one, so the translation never fired and the task failed unretried. Which
family a given SDK resolves to is an environment detail, so the safe
assumption is that any of them may raise.
"""

from __future__ import annotations

__all__ = ["TRANSIENT_TRANSPORT_ERRORS", "TRANSPORT_MODULES"]

import importlib

# Both the httpx-level names and the httpcore-level ones. httpx normally
# remaps its core's exceptions to its own, but does not guarantee it on
# every path, and a traceback from a real failure showed the core class
# as the direct cause of the httpx one.
TRANSPORT_MODULES = ("httpx", "httpx2", "httpcore", "httpcore2")

# Enumerated rather than catching TransportError wholesale, because two of
# its children are not transient and must not be retried:
# LocalProtocolError means we built a bad request, and UnsupportedProtocol
# is deterministic. Retrying either just repeats the same failure with
# backoff.
_TRANSIENT_NAMES = (
    "TimeoutException",  # read/connect/pool/write timeouts
    "NetworkError",  # connect/read/write/close errors
    "RemoteProtocolError",  # peer closed mid-response
    "ProxyError",
)


def _collect() -> tuple[type[BaseException], ...]:
    found: list[type[BaseException]] = []
    for module_name in TRANSPORT_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in _TRANSIENT_NAMES:
            exc = getattr(module, name, None)
            if isinstance(exc, type) and issubclass(exc, BaseException):
                found.append(exc)
    return tuple(found)


TRANSIENT_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = _collect()
