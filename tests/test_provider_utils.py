"""Tests for the provider utilities module."""

from unittest.mock import MagicMock, patch

import pytest
from llmproc.providers.utils import get_context_window_size


class TestContextWindowSize:
    """Tests for the context window size function."""

    def test_get_context_window_size_exact_match(self):
        """Test getting window size with exact model match."""
        window_sizes = {
            "gpt-4o-mini": 8192,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 16385,
        }

        assert get_context_window_size("gpt-4o-mini", window_sizes) == 8192
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

        assert get_context_window_size("unknown-model", window_sizes, default_size=50000) == 50000
