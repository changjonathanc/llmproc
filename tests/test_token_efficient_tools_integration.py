"""Integration tests for token-efficient tool use feature.

This file follows the standardized API test patterns.
It uses fixtures from conftest_api.py and the standard Arrange-Act-Assert pattern.
"""

import os
import time

import pytest

# Register conftest_api as a plugin for this test file
pytest_plugins = ["tests.conftest_api"]

from llmproc import LLMProgram


@pytest.fixture
async def token_efficient_tools_from_sdk():
    """Creates a process with token-efficient tools using the Python SDK.

    This fixture provides an isolated process instance with token-efficient tools
    enabled, created programmatically using the Python SDK instead of TOML.

    Uses function scope for test isolation and the program.start() pattern.

    Yields:
        LLMProcess: A started process instance with token-efficient tools enabled
    """
    # Create a program using the Python SDK with token-efficient tools enabled
    program = (
        LLMProgram(
            model_name="claude-3-7-sonnet-20250219",  # Required for token-efficient tools
            provider="anthropic",
            system_prompt="You are a helpful assistant with access to tools.",
            parameters={
                "max_tokens": 4096,
                "temperature": 0.7,
            },
        )
        .register_tools(["calculator"])  # Enable calculator tool for testing
        .enable_token_efficient_tools()  # Enable token-efficient tools feature
    )

    # Start the process using the standard pattern
    process = await program.start()
    yield process


@pytest.mark.llm_api
@pytest.mark.extended_api
class TestTokenEfficientToolsIntegration:
    """Integration test suite for token-efficient tool use with actual API calls.

    These tests verify that the token-efficient tools feature works correctly
    with real API calls to Claude 3.7.
    """

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key found")
    @pytest.mark.asyncio
    async def test_fixture_calculator_tool_works(self, claude_process_with_token_efficient_tools):
        """Test that calculator tool works with token-efficient mode."""
        # Arrange
        process = claude_process_with_token_efficient_tools
        start_time = time.time()

        # Act
        result = await process.run("What is the square root of 256? Use the calculator tool.")

        # Assert
        # Verify the correct answer was calculated
        last_message = process.get_last_message()
        assert "16" in last_message, f"Expected calculator result '16' in message: {last_message}"

        # Check for usage information
        assert result.api_calls > 0

        # Verify test completes within reasonable time
        duration = time.time() - start_time
        assert duration < 20.0, f"Test took too long: {duration:.2f}s"

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key found")
    @pytest.mark.asyncio
    async def test_sdk_config_has_proper_headers(self, token_efficient_tools_from_sdk):
        """Test that token-efficient tools in SDK config sets up the proper headers."""
        # Arrange
        process = token_efficient_tools_from_sdk

        # Act - Nothing to do here, just verify the configuration

        # Assert - Check headers are properly configured
        assert "extra_headers" in process.api_params
        assert "anthropic-beta" in process.api_params["extra_headers"]
        assert "token-efficient-tools" in process.api_params["extra_headers"]["anthropic-beta"]

        # Check that calculator tool is enabled
        assert "calculator" in [tool["name"] for tool in process.tools]

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key found")
    @pytest.mark.asyncio
    async def test_sdk_calculator_tool_works(self, token_efficient_tools_from_sdk):
        """Test that calculator tool from SDK config works with token-efficient mode."""
        # Arrange
        process = token_efficient_tools_from_sdk
        start_time = time.time()

        # Act
        result = await process.run("What is the square root of 256? Use the calculator tool.")

        # Assert
        # Verify the correct answer was calculated
        last_message = process.get_last_message()
        assert "16" in last_message, f"Expected calculator result '16' in message: {last_message}"

        # Check for usage information
        assert result.api_calls > 0

        # Verify test completes within reasonable time
        duration = time.time() - start_time
        assert duration < 20.0, f"Test took too long: {duration:.2f}s"
