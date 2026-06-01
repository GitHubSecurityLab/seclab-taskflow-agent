# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for exception-construction helpers."""

from seclab_taskflow_agent.error_utils import error_with_message
from seclab_taskflow_agent.sdk.errors import BackendTimeoutError


def test_error_with_message_preserves_type_and_message():
    exc = error_with_message(BackendTimeoutError, "timed out")

    assert isinstance(exc, BackendTimeoutError)
    assert str(exc) == "timed out"
