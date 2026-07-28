# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for TmpEnv context manager in env_utils."""

import os
import pytest
from seclab_taskflow_agent.env_utils import TmpEnv


class TestTmpEnv:
    """Test suite for TmpEnv context manager."""

    def test_tmpenv_restores_existing_env_var(self):
        """Test that TmpEnv restores an existing environment variable to its original value."""
        # Setup: Set an existing environment variable
        original_value = "original_value"
        os.environ["TEST_VAR_EXISTING"] = original_value
        
        try:
            # Use TmpEnv to temporarily change the value
            with TmpEnv({"TEST_VAR_EXISTING": "temporary_value"}):
                assert os.environ["TEST_VAR_EXISTING"] == "temporary_value"
            
            # After exiting context, should restore to original value
            assert os.environ["TEST_VAR_EXISTING"] == original_value
        finally:
            # Cleanup
            del os.environ["TEST_VAR_EXISTING"]

    def test_tmpenv_removes_newly_added_env_var(self):
        """Test that TmpEnv removes an environment variable that didn't exist before."""
        # Ensure the variable doesn't exist
        if "TEST_VAR_NEW" in os.environ:
            del os.environ["TEST_VAR_NEW"]
        
        # Use TmpEnv to add a new environment variable
        with TmpEnv({"TEST_VAR_NEW": "new_value"}):
            assert os.environ["TEST_VAR_NEW"] == "new_value"
        
        # After exiting context, the variable should be removed
        assert "TEST_VAR_NEW" not in os.environ

    def test_tmpenv_handles_multiple_vars(self):
        """Test that TmpEnv handles multiple environment variables correctly."""
        # Setup: One existing, one new
        os.environ["TEST_VAR_MULTI_EXISTING"] = "original"
        if "TEST_VAR_MULTI_NEW" in os.environ:
            del os.environ["TEST_VAR_MULTI_NEW"]
        
        try:
            with TmpEnv({
                "TEST_VAR_MULTI_EXISTING": "changed",
                "TEST_VAR_MULTI_NEW": "added"
            }):
                assert os.environ["TEST_VAR_MULTI_EXISTING"] == "changed"
                assert os.environ["TEST_VAR_MULTI_NEW"] == "added"
            
            # After exit: existing restored, new removed
            assert os.environ["TEST_VAR_MULTI_EXISTING"] == "original"
            assert "TEST_VAR_MULTI_NEW" not in os.environ
        finally:
            # Cleanup
            if "TEST_VAR_MULTI_EXISTING" in os.environ:
                del os.environ["TEST_VAR_MULTI_EXISTING"]

    def test_tmpenv_restores_on_exception(self):
        """Test that TmpEnv restores environment even when exception occurs."""
        os.environ["TEST_VAR_EXCEPTION"] = "original"
        
        try:
            with pytest.raises(ValueError):
                with TmpEnv({"TEST_VAR_EXCEPTION": "temporary"}):
                    assert os.environ["TEST_VAR_EXCEPTION"] == "temporary"
                    raise ValueError("Test exception")
            
            # Should still restore even after exception
            assert os.environ["TEST_VAR_EXCEPTION"] == "original"
        finally:
            # Cleanup
            del os.environ["TEST_VAR_EXCEPTION"]

    def test_tmpenv_empty_dict(self):
        """Test that TmpEnv handles empty dict correctly."""
        # Should not raise any errors
        with TmpEnv({}):
            pass
        
        # Environment should be unchanged
        assert True  # If we got here, no exception was raised

    def test_tmpenv_with_context(self):
        """Test that TmpEnv works with context parameter for template rendering."""
        os.environ["TEST_VAR_CONTEXT"] = "original"
        
        try:
            # TmpEnv should accept context parameter (even if we don't use it in this test)
            with TmpEnv({"TEST_VAR_CONTEXT": "new"}, context={"key": "value"}):
                assert os.environ["TEST_VAR_CONTEXT"] == "new"
            
            assert os.environ["TEST_VAR_CONTEXT"] == "original"
        finally:
            # Cleanup
            del os.environ["TEST_VAR_CONTEXT"]
