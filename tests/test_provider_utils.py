"""Tests for the provider utilities module."""

from unittest.mock import MagicMock, patch

import pytest

from llmproc.providers.utils import (
    get_context_window_size,
    safe_callback,
)


class TestSafeCallback:
    """Tests for the provider utilities safe callback function."""

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


class TestContextWindowSize:
    """Tests for the context window size function."""

    def test_get_context_window_size_exact_match(self):
        """Test getting window size with exact model match."""
        window_sizes = {
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 16385,
        }

        assert get_context_window_size("gpt-4", window_sizes) == 8192
        assert get_context_window_size("gpt-3.5-turbo", window_sizes) == 16385

    def test_get_context_window_size_prefix_match(self):
        """Test getting window size with prefix match."""
        window_sizes = {
            "claude-3-": 200000,
            "gemini-1.5": 1000000,
        }

        assert get_context_window_size("claude-3-opus", window_sizes) == 200000
        assert get_context_window_size("gemini-1.5-flash", window_sizes) == 1000000

    def test_get_context_window_size_with_version_number(self):
        """Test getting window size with version in name."""
        window_sizes = {
            "claude-3": 200000,
        }

        assert get_context_window_size("claude-3-20240229", window_sizes) == 200000

    def test_get_context_window_size_default(self):
        """Test default window size for unknown models."""
        window_sizes = {
            "gpt-4": 8192,
        }

        assert get_context_window_size("unknown-model", window_sizes) == 100000

    def test_get_context_window_size_custom_default(self):
        """Test custom default window size."""
        window_sizes = {
            "gpt-4": 8192,
        }

        assert (
            get_context_window_size("unknown-model", window_sizes, default_size=50000)
            == 50000
        )
