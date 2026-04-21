# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for env_utils: swap_env and TmpEnv with globals context."""

import os

import pytest

from seclab_taskflow_agent.env_utils import TmpEnv, swap_env


class TestSwapEnv:
    """Tests for swap_env template rendering."""

    def test_plain_string_unchanged(self):
        assert swap_env("no templates here") == "no templates here"

    def test_env_function_works(self):
        os.environ["TEST_SWAP_ENV_VAR"] = "hello"
        try:
            assert swap_env('{{ env("TEST_SWAP_ENV_VAR") }}') == "hello"
        finally:
            del os.environ["TEST_SWAP_ENV_VAR"]

    def test_globals_with_context(self):
        result = swap_env(
            "key-{{ globals.ghsa_id }}",
            context={"globals": {"ghsa_id": "GHSA-1234"}},
        )
        assert result == "key-GHSA-1234"

    def test_globals_without_context_raises(self):
        with pytest.raises(LookupError):
            swap_env("{{ globals.missing }}")

    def test_context_cannot_override_env_helper(self):
        """Passing an 'env' key in context must not shadow the env() function."""
        os.environ["TEST_SWAP_RESERVED"] = "works"
        try:
            result = swap_env(
                '{{ env("TEST_SWAP_RESERVED") }}',
                context={"env": "should be filtered"},
            )
            assert result == "works"
        finally:
            del os.environ["TEST_SWAP_RESERVED"]

    def test_no_context_backward_compat(self):
        assert swap_env("plain") == "plain"


class TestTmpEnv:
    """Tests for TmpEnv context manager with globals."""

    def test_globals_rendered_in_env_block(self):
        env = {"MY_KEY": "pvr-{{ globals.ghsa }}"}
        ctx = {"globals": {"ghsa": "GHSA-5678"}}
        with TmpEnv(env, context=ctx):
            assert os.environ["MY_KEY"] == "pvr-GHSA-5678"
        assert "MY_KEY" not in os.environ

    def test_env_function_still_works_in_tmpenv(self):
        os.environ["SOURCE_VAR"] = "value"
        try:
            env = {"DEST_VAR": '{{ env("SOURCE_VAR") }}'}
            with TmpEnv(env):
                assert os.environ["DEST_VAR"] == "value"
        finally:
            del os.environ["SOURCE_VAR"]

    def test_tmpenv_restores_original(self):
        os.environ["RESTORE_TEST"] = "original"
        env = {"RESTORE_TEST": "overwritten"}
        with TmpEnv(env):
            assert os.environ["RESTORE_TEST"] == "overwritten"
        assert os.environ["RESTORE_TEST"] == "original"
        del os.environ["RESTORE_TEST"]

    def test_tmpenv_rollback_on_error(self):
        """Partial env modification is rolled back if swap_env raises."""
        env = {"GOOD_KEY": "value", "BAD_KEY": "{{ globals.missing }}"}
        with pytest.raises(LookupError), TmpEnv(env):
            pass
        assert "GOOD_KEY" not in os.environ
        assert "BAD_KEY" not in os.environ
