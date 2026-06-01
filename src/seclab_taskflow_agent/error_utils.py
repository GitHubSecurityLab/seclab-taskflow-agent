# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Helpers for constructing exceptions without inline raise messages."""

from __future__ import annotations

__all__ = ["error_with_message"]

from typing import TypeVar

ExcT = TypeVar("ExcT", bound=BaseException)


def error_with_message(exc_type: type[ExcT], message: str, /) -> ExcT:
    """Return *exc_type* initialised with *message*."""
    return exc_type(message)
