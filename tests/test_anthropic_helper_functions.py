"""Tests for the Anthropic Process Executor helper functions."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from llmproc.providers.anthropic_utils import (
    add_token_efficient_header_if_needed,
)
from llmproc.providers.constants import ANTHROPIC_PROVIDERS
from llmproc.providers.utils import safe_callback


class TestAnthropicHelperFunctions:
    """Tests for helper functions in the Anthropic Process Executor module."""

    def test_add_token_efficient_header_empty_headers(self):
        """Test adding token-efficient header to empty headers."""
        process = MagicMock()
        process.provider = "anthropic"
        process.model_name = "claude-3-7-sonnet-20250219"

        headers = {}
        result = add_token_efficient_header_if_needed(process, headers)

        assert "anthropic-beta" in result
        assert result["anthropic-beta"] == "token-efficient-tools-2025-02-19"

    def test_add_token_efficient_header_existing_headers(self):
        """Test adding token-efficient header to existing headers."""
        process = MagicMock()
        process.provider = "anthropic"
        process.model_name = "claude-3-7-sonnet-20250219"

        headers = {"anthropic-beta": "existing-feature"}
        result = add_token_efficient_header_if_needed(process, headers)

        assert "anthropic-beta" in result
        assert "existing-feature,token-efficient-tools-2025-02-19" == result["anthropic-beta"]

    def test_add_token_efficient_header_already_present(self):
        """Test not duplicating token-efficient header if already present."""
        process = MagicMock()
        process.provider = "anthropic"
        process.model_name = "claude-3-7-sonnet-20250219"

        headers = {"anthropic-beta": "token-efficient-tools-2025-02-19"}
        result = add_token_efficient_header_if_needed(process, headers)

        assert "anthropic-beta" in result
        assert result["anthropic-beta"] == "token-efficient-tools-2025-02-19"

    def test_add_token_efficient_header_non_claude_37(self):
        """Test that header is not added for non-Claude 3.7 models."""
        process = MagicMock()
        process.provider = "anthropic"
        process.model_name = "claude-3-5-sonnet-20241022"

        headers = {}
        result = add_token_efficient_header_if_needed(process, headers)

        assert not headers  # Original headers unchanged
        assert result == headers  # Result headers should be empty

    def test_safe_callback_successful_execution(self):
        """Test successful callback execution."""
        callback_fn = MagicMock()
        safe_callback(callback_fn, "arg1", "arg2", callback_name="test_callback")

        callback_fn.assert_called_once_with("arg1", "arg2")

    @patch("llmproc.providers.utils.logger")
    def test_safe_callback_handles_exception(self, mock_logger):
        """Test that exceptions in callbacks are caught and logged."""
        callback_fn = MagicMock(side_effect=Exception("Test error"))

        # This should not raise an exception
        safe_callback(callback_fn, "arg1", callback_name="test_callback")

        callback_fn.assert_called_once_with("arg1")
        mock_logger.warning.assert_called_once()
        assert "Error in test_callback callback" in mock_logger.warning.call_args[0][0]

    def test_safe_callback_none_callback(self):
        """Test handling None callback."""
        # Should not raise an exception
        safe_callback(None, "arg1", callback_name="test_callback")
