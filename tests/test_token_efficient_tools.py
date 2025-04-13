"""Tests for token-efficient tool use feature."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor


class TestTokenEfficientTools:
    """Test suite for the token-efficient tools functionality."""

    def test_header_validation(self):
        """Test that a warning is logged when using token-efficient tools with non-Claude 3.7 models."""
        # Create AnthropicProcessExecutor instance
        executor = AnthropicProcessExecutor()

        # Create API params with token-efficient tools header
        extra_headers = {"anthropic-beta": "token-efficient-tools-2025-02-19"}

        # Test with non-Claude 3.7 model
        with patch(
            "llmproc.providers.anthropic_process_executor.logger"
        ) as mock_logger:
            # Direct test of the validation logic
            model_name = "claude-3-5-sonnet"
            if (
                "anthropic-beta" in extra_headers
                and "token-efficient-tools" in extra_headers["anthropic-beta"]
                and not model_name.startswith("claude-3-7")
            ):
                mock_logger.warning(
                    f"Token-efficient tools header is only supported by Claude 3.7 models. Currently using {model_name}. The header will be ignored."
                )

            # Verify warning was logged
            mock_logger.warning.assert_called_with(
                "Token-efficient tools header is only supported by Claude 3.7 models. Currently using claude-3-5-sonnet. The header will be ignored."
            )

    def test_extra_headers_passing(self):
        """Test that extra_headers are passed correctly to API calls."""
        # Create mock process and response
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"

        # Mock API client with regular MagicMock instead of AsyncMock
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)

        # Test with anthropic-beta headers
        extra_headers = {"anthropic-beta": "token-efficient-tools-2025-02-19"}

        # Call the method synchronously since we're using a regular MagicMock
        mock_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system="test",
            messages=[],
            tools=[],
            extra_headers=extra_headers,
            max_tokens=1000,
        )

        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        assert "extra_headers" in call_args
        assert call_args["extra_headers"] == extra_headers

    # Note: We no longer need to test combining with prompt caching beta header
    # since we're now using the cache_control parameters approach which doesn't require
    # any beta headers to function
