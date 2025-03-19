"""Tests for error handling utilities."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from llmproc.utils.error_handling import (
    AuthenticationError,
    LLMProcError,
    ProviderAPIError,
    RateLimitError,
    classify_provider_error,
    dump_api_error,
)


class TestErrorClasses:
    """Test error classes."""

    def test_llmproc_error(self):
        """Test LLMProcError class."""
        error = LLMProcError("Test error", provider="test")
        assert str(error) == "Test error"
        assert error.provider == "test"
        assert error.details == {}

    def test_provider_api_error(self):
        """Test ProviderAPIError class."""
        error = ProviderAPIError("API error", provider="anthropic", status_code=400)
        assert str(error) == "API error"
        assert error.provider == "anthropic"
        assert error.status_code == 400

    def test_authentication_error(self):
        """Test AuthenticationError class."""
        error = AuthenticationError("Auth error", provider="openai", status_code=401)
        assert str(error) == "Auth error"
        assert error.provider == "openai"
        assert error.status_code == 401
        assert isinstance(error, ProviderAPIError)

    def test_rate_limit_error(self):
        """Test RateLimitError class."""
        error = RateLimitError("Rate limit error", provider="anthropic", status_code=429)
        assert str(error) == "Rate limit error"
        assert error.provider == "anthropic"
        assert error.status_code == 429
        assert isinstance(error, ProviderAPIError)


class TestErrorUtils:
    """Test error utility functions."""

    def test_dump_api_error(self, tmp_path):
        """Test dump_api_error function."""
        # Set up test data
        error = ValueError("Test error")
        provider = "test_provider"
        model_name = "test_model"
        system_prompt = "You are a helpful assistant"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        api_params = {"temperature": 0.7, "max_tokens": 1000}
        tools = [{"name": "calculator", "description": "A calculator tool"}]

        # Mock Path.mkdir to use tmp_path
        with patch("pathlib.Path.mkdir") as mock_mkdir, patch(
            "datetime.datetime.now"
        ) as mock_now, patch("pathlib.Path") as mock_path:
            # Set up mocks
            mock_now.return_value.strftime.return_value = "20250319_120000"
            mock_path.return_value = tmp_path
            mock_path.side_effect = lambda p: tmp_path / p if isinstance(p, str) else p
            mock_mkdir.return_value = None

            # Call function
            result = dump_api_error(
                error,
                provider,
                model_name,
                system_prompt,
                messages,
                api_params,
                tools,
                context={"test": "context"},
            )

            # Read the dump file
            dump_file = list(tmp_path.glob("*.json"))[0]
            with open(dump_file, "r") as f:
                dump_data = json.load(f)

            # Check structure
            assert "error" in dump_data
            assert "api_params" in dump_data
            assert "tools" in dump_data
            assert "context" in dump_data

            # Check content
            assert dump_data["error"]["message"] == "Test error"
            assert dump_data["error"]["type"] == "ValueError"
            assert dump_data["api_params"]["provider"] == "test_provider"
            assert dump_data["api_params"]["model"] == "test_model"
            assert dump_data["api_params"]["temperature"] == 0.7
            assert dump_data["tools"]["count"] == 1

    def test_classify_anthropic_auth_error(self):
        """Test classify_provider_error for Anthropic authentication error."""
        original_error = Exception("Invalid API key")
        original_error.status_code = 401

        error = classify_provider_error(original_error, "anthropic")

        assert isinstance(error, AuthenticationError)
        assert error.provider == "anthropic"
        assert error.status_code == 401
        assert "Anthropic authentication error" in str(error)

    def test_classify_openai_rate_limit_error(self):
        """Test classify_provider_error for OpenAI rate limit error."""
        original_error = Exception("Rate limit exceeded")
        original_error.status_code = 429

        error = classify_provider_error(original_error, "openai")

        assert isinstance(error, RateLimitError)
        assert error.provider == "openai"
        assert error.status_code == 429
        assert "OpenAI rate limit exceeded" in str(error)

    def test_classify_unknown_provider_error(self):
        """Test classify_provider_error for unknown provider."""
        original_error = Exception("Unknown error")

        error = classify_provider_error(original_error, "unknown_provider")

        assert isinstance(error, ProviderAPIError)
        assert error.provider == "unknown_provider"
        assert "unknown_provider API error" in str(error)