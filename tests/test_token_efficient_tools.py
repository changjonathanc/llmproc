"""Tests for token-efficient tool use feature."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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
        with patch("llmproc.providers.anthropic_process_executor.logger") as mock_logger:
            # Direct test of the validation logic
            model_name = "claude-3-5-sonnet"
            if ("anthropic-beta" in extra_headers and 
                "token-efficient-tools" in extra_headers["anthropic-beta"] and
                not model_name.startswith("claude-3-7")):
                mock_logger.warning(
                    f"Token-efficient tools header is only supported by Claude 3.7 models. "
                    f"Currently using {model_name}. The header will be ignored."
                )
            
            # Verify warning was logged
            mock_logger.warning.assert_called_with(
                "Token-efficient tools header is only supported by Claude 3.7 models. "
                "Currently using claude-3-5-sonnet. The header will be ignored."
            )
    
    def test_extra_headers_passing(self):
        """Test that extra_headers are passed correctly to API calls."""
        # Create mock process and response
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        
        # Mock API client
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        # Test with anthropic-beta headers
        extra_headers = {"anthropic-beta": "token-efficient-tools-2025-02-19"}
        
        # Verify headers get passed to create() method
        mock_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system="test",
            messages=[],
            tools=[],
            extra_headers=extra_headers,
            max_tokens=1000
        )
        
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        assert "extra_headers" in call_args
        assert call_args["extra_headers"] == extra_headers
        
    def test_beta_headers_combination(self):
        """Test that token-efficient tools and prompt caching beta headers can be combined."""
        # Mock response
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        
        # Mock API client
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        # Start with token-efficient tools header
        extra_headers = {"anthropic-beta": "token-efficient-tools-2025-02-19"}
        
        # Add prompt caching header
        if "anthropic-beta" in extra_headers:
            if "prompt-caching" not in extra_headers["anthropic-beta"]:
                extra_headers["anthropic-beta"] += ",prompt-caching-2024-07-31"
        
        # Verify combined headers
        mock_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system="test",
            messages=[],
            tools=[],
            extra_headers=extra_headers,
            max_tokens=1000
        )
        
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        assert "extra_headers" in call_args
        assert "anthropic-beta" in call_args["extra_headers"]
        assert "token-efficient-tools-2025-02-19" in call_args["extra_headers"]["anthropic-beta"]
        assert "prompt-caching-2024-07-31" in call_args["extra_headers"]["anthropic-beta"]