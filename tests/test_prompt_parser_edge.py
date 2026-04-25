# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Edge-case tests for the legacy prompt parser."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from seclab_taskflow_agent.available_tools import AvailableTools
from seclab_taskflow_agent.prompt_parser import parse_prompt_args


def _tools() -> AvailableTools:
    return MagicMock(spec=AvailableTools)


class TestParsePromptArgs:
    """Tests for parse_prompt_args edge cases."""

    @patch("sys.argv", ["main"])
    def test_none_prompt_returns_defaults(self):
        """None prompt causes argparse to read sys.argv; result still has 6 elements."""
        result = parse_prompt_args(_tools(), None)
        assert len(result) == 6
        # When None is passed, argparse parses sys.argv[1:] (pytest args),
        # so personality and taskflow remain None
        p, t, list_flag, g, prompt, help_msg = result
        assert p is None
        assert t is None

    @patch("sys.argv", ["main"])
    def test_empty_string_returns_defaults(self):
        """Empty string prompt returns default values."""
        result = parse_prompt_args(_tools(), "")
        assert len(result) == 6
        p, t, list_flag, g, prompt, help_msg = result
        assert p is None
        assert t is None

    def test_personality_flag(self):
        """-p flag sets the personality."""
        p, t, list_flag, g, prompt, _ = parse_prompt_args(
            _tools(), "-p my.personality hello world"
        )
        assert p == "my.personality"
        assert t is None
        assert prompt == "hello world"

    def test_taskflow_flag(self):
        """-t flag sets the taskflow."""
        p, t, list_flag, g, prompt, _ = parse_prompt_args(
            _tools(), "-t my.taskflow do stuff"
        )
        assert t == "my.taskflow"
        assert p is None
        assert prompt == "do stuff"

    def test_list_models_flag(self):
        """-l flag sets list_models to True."""
        p, t, list_flag, g, prompt, _ = parse_prompt_args(_tools(), "-l")
        assert list_flag is True
        assert p is None
        assert t is None

    def test_invalid_global_format_returns_none_tuple(self):
        """-g with no = returns the None/error tuple."""
        result = parse_prompt_args(_tools(), "-g badformat")
        p, t, list_flag, g, prompt, help_msg = result
        assert p is None
        assert t is None
        assert list_flag is None
        assert g is None

    def test_mutual_exclusivity_p_and_t(self):
        """-p and -t together triggers SystemExit → None tuple."""
        result = parse_prompt_args(_tools(), "-p foo -t bar")
        p, t, list_flag, g, prompt, help_msg = result
        # argparse raises SystemExit(2) which is caught → None tuple
        assert p is None
        assert t is None
        assert list_flag is None
        assert g is None

    def test_prompt_remainder_collected(self):
        """Remaining text after flags is collected as prompt."""
        _, _, _, _, prompt, _ = parse_prompt_args(
            _tools(), "-p my.personality tell me a joke"
        )
        assert prompt == "tell me a joke"

    def test_return_tuple_always_has_six_elements(self):
        """Return value always has exactly 6 elements."""
        # Success case
        result = parse_prompt_args(_tools(), "-p my.personality hello")
        assert len(result) == 6

        # Error case
        result = parse_prompt_args(_tools(), "-p foo -t bar")
        assert len(result) == 6

        # None case
        result = parse_prompt_args(_tools(), None)
        assert len(result) == 6

    def test_global_variable_valid(self):
        """-g KEY=VALUE correctly parses global variables."""
        _, _, _, g, _, _ = parse_prompt_args(
            _tools(), "-p my.personality -g fruit=apple hello"
        )
        assert g == {"fruit": "apple"}

    def test_multiple_globals(self):
        """Multiple -g flags are all captured."""
        _, _, _, g, _, _ = parse_prompt_args(
            _tools(), "-p my.personality -g fruit=apple -g color=red"
        )
        assert g == {"fruit": "apple", "color": "red"}

    def test_global_value_with_equals(self):
        """-g with value containing = uses only first = as delimiter."""
        _, _, _, g, _, _ = parse_prompt_args(
            _tools(), "-p my.personality -g url=https://x.com?a=1"
        )
        assert g == {"url": "https://x.com?a=1"}
