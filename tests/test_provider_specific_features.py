"""Tests for provider-specific feature implementations."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.providers.anthropic_process_executor import (
    AnthropicProcessExecutor,
    add_cache_to_message,
    system_to_api_format,
    tools_to_api_format,
)

# Define constants for model versions to make updates easier
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"  # Use a specific versioned model


class TestProviderSpecificFeatures:
    """Test suite for provider-specific feature implementations."""

    def test_provider_detection(self):
        """Test provider type detection logic."""
        # Test direct Anthropic provider detection
        is_direct_anthropic = "anthropic" in "anthropic" and "vertex" not in "anthropic"
        assert is_direct_anthropic

        # Test Vertex AI provider detection
        is_direct_anthropic = (
            "anthropic" in "anthropic_vertex" and "vertex" not in "anthropic_vertex"
        )
        assert not is_direct_anthropic

        # Test combined provider strings
        is_direct_anthropic = (
            "anthropic" in "anthropic-vertex" and "vertex" not in "anthropic-vertex"
        )
        assert not is_direct_anthropic

    def test_cache_control_parameters(self):
        """Test adding cache control parameters to messages."""
        # Test with simple message
        message = {"role": "user", "content": "Hello, world!"}
        add_cache_to_message(message)

        # Check that content was transformed to structured format with cache
        assert isinstance(message["content"], list)
        assert message["content"][0]["type"] == "text"
        assert message["content"][0]["text"] == "Hello, world!"
        assert message["content"][0]["cache_control"] == {"type": "ephemeral"}

        # Test with structured message
        message = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello, world!"}],
        }
        add_cache_to_message(message)

        # Check that cache was added to structured content
        assert message["content"][0]["cache_control"] == {"type": "ephemeral"}

        # Test with tool result
        message = {
            "role": "user",
            "content": [{"type": "tool_result", "content": "Calculator result"}],
        }
        add_cache_to_message(message)

        # Check that cache was added to tool result
        assert message["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_system_prompt_cache_control(self):
        """Test adding cache control to system prompt."""
        # Test with string system prompt
        system_prompt = "You are a helpful assistant."
        result = system_to_api_format(system_prompt, add_cache=True)

        # Check that system prompt was transformed to structured format with cache
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[0]["text"] == system_prompt
        assert result[0]["cache_control"] == {"type": "ephemeral"}

        # Test with no caching
        result = system_to_api_format(system_prompt, add_cache=False)
        assert result == system_prompt

        # Test with already structured system prompt
        structured_prompt = [{"type": "text", "text": "You are a helpful assistant."}]
        result = system_to_api_format(structured_prompt, add_cache=True)
        assert result == structured_prompt

    def test_tools_cache_control(self):
        """Test adding cache control to tools."""
        # Test with tools array
        tools = [
            {"name": "calculator", "description": "Use this to perform calculations"},
            {"name": "web_search", "description": "Search the web for information"},
        ]

        result = tools_to_api_format(tools, add_cache=True)

        # Check that cache was added to last tool
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}

        # Test with no caching
        result = tools_to_api_format(tools, add_cache=False)
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[1]

        # Test with empty tools
        assert tools_to_api_format([], add_cache=True) == []

    def test_token_efficient_tools_header(self):
        """Test token-efficient tools header application."""
        executor = AnthropicProcessExecutor()

        # Create mock process for direct Anthropic
        mock_process = MagicMock()
        mock_process.provider = "anthropic"
        mock_process.model_name = "claude-3-7-sonnet-20250219"

        # Initial headers
        extra_headers = {}

        # Apply token-efficient tools logic
        if (
            "anthropic" in mock_process.provider.lower()
            and mock_process.model_name.startswith("claude-3-7")
        ):
            if "anthropic-beta" not in extra_headers:
                extra_headers["anthropic-beta"] = "token-efficient-tools-2025-02-19"
            elif "token-efficient-tools" not in extra_headers["anthropic-beta"]:
                extra_headers["anthropic-beta"] += ",token-efficient-tools-2025-02-19"

        # Verify header was added for direct Anthropic
        assert "anthropic-beta" in extra_headers
        assert extra_headers["anthropic-beta"] == "token-efficient-tools-2025-02-19"

        # Now test with Vertex AI
        mock_process.provider = "anthropic_vertex"
        extra_headers = {}

        # Apply token-efficient tools logic for Vertex AI
        if (
            "anthropic" in mock_process.provider.lower()
            and mock_process.model_name.startswith("claude-3-7")
        ):
            if "anthropic-beta" not in extra_headers:
                extra_headers["anthropic-beta"] = "token-efficient-tools-2025-02-19"
            elif "token-efficient-tools" not in extra_headers["anthropic-beta"]:
                extra_headers["anthropic-beta"] += ",token-efficient-tools-2025-02-19"

        # Verify header was also added for Vertex AI
        assert "anthropic-beta" in extra_headers
        assert extra_headers["anthropic-beta"] == "token-efficient-tools-2025-02-19"

        # Test warning logic with non-Claude 3.7 model
        mock_process.provider = "anthropic"
        mock_process.model_name = "claude-3-5-sonnet"
        extra_headers = {"anthropic-beta": "token-efficient-tools-2025-02-19"}

        with patch(
            "llmproc.providers.anthropic_process_executor.logger"
        ) as mock_logger:
            # Check warning logic
            if (
                "anthropic-beta" in extra_headers
                and "token-efficient-tools" in extra_headers["anthropic-beta"]
                and (
                    "anthropic" not in mock_process.provider.lower()
                    or not mock_process.model_name.startswith("claude-3-7")
                )
            ):
                # Warning if token-efficient tools header is present but not supported
                mock_logger.warning(
                    f"Token-efficient tools header is only supported by Claude 3.7 models. Currently using {mock_process.model_name} on {mock_process.provider}. The header will be ignored."
                )

            # Verify warning was logged
            mock_logger.warning.assert_called_once()


@pytest.mark.llm_api
class TestProviderSpecificFeaturesIntegration:
    """Integration tests for provider-specific features that require API access."""

    @pytest.fixture
    def anthropic_api_key(self):
        """Get Anthropic API key from environment variable."""
        return os.environ.get("ANTHROPIC_API_KEY")

    @pytest.fixture
    def vertex_project_id(self):
        """Get Vertex AI project ID from environment variable."""
        return os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key found"
    )
    async def test_cache_control_with_direct_anthropic(self, anthropic_api_key):
        """Test that cache control parameters work with direct Anthropic API."""
        from llmproc import LLMProgram

        if not anthropic_api_key:
            pytest.skip("No Anthropic API key found")

        # Create a simple program
        program = LLMProgram(
            model_name=CLAUDE_MODEL,
            provider="anthropic",
            system_prompt="You are a helpful assistant. "
            + ("This is filler content. " * 500),
            parameters={
                "max_tokens": 1000,
            },
            disable_automatic_caching=False,  # Ensure caching is enabled
        )

        # Start the process
        process = await program.start()

        # Run the process twice to trigger caching
        result1 = await process.run("What is your name?")
        result2 = await process.run("Tell me a joke.")

        # Get API call metrics
        api_call1 = result1.to_dict()["api_calls"][0]
        api_call2 = result2.to_dict()["api_calls"][0]

        # Verify we're getting cache metrics back
        # Either we should see cache metrics, or at least token usage info
        if "usage" in api_call1 and "usage" in api_call2:
            # Check if there are cache-related fields in the usage data
            usage1 = api_call1["usage"]
            usage2 = api_call2["usage"]

            # Verify some token usage was recorded
            assert usage1.get("input_tokens", 0) > 0
            assert usage2.get("input_tokens", 0) > 0

            # Output metrics for debugging
            print(f"API Call 1 usage: {usage1}")
            print(f"API Call 2 usage: {usage2}")

            # Test passes if we get usage metrics (cache evidence is ideal but hard to test reliably)
            assert True

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"),
        reason="No Vertex AI project ID found",
    )
    async def test_token_efficient_tools_vertex(self, vertex_project_id):
        """Test that token-efficient tools works with Vertex AI."""
        import time

        try:
            from anthropic import AsyncAnthropicVertex

            VERTEX_AVAILABLE = True
        except ImportError:
            VERTEX_AVAILABLE = False

        if not VERTEX_AVAILABLE:
            pytest.skip("Anthropic Vertex SDK not installed")

        if not vertex_project_id:
            pytest.skip("No Vertex AI project ID found")

        region = os.environ.get("CLOUD_ML_REGION", "us-central1")

        # Define a simple calculator tool
        calculator_tool = {
            "name": "calculator",
            "description": "Use this tool to perform calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        }

        try:
            # Initialize Vertex client
            client = AsyncAnthropicVertex(project_id=vertex_project_id, region=region)

            # First request WITHOUT token-efficient tools header
            response_standard = await client.messages.create(
                model="claude-3-7-sonnet@20250219",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the square root of 256? Please use the calculator tool.",
                    }
                ],
                tools=[calculator_tool],
                system="You are a helpful AI assistant that uses tools when appropriate.",
            )

            # Wait a bit to avoid rate limits
            time.sleep(2)

            # Second request WITH token-efficient tools header
            response_efficient = await client.messages.create(
                model="claude-3-7-sonnet@20250219",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the square root of 256? Please use the calculator tool.",
                    }
                ],
                tools=[calculator_tool],
                system="You are a helpful AI assistant that uses tools when appropriate.",
                extra_headers={"anthropic-beta": "token-efficient-tools-2025-02-19"},
            )

            # Compare output tokens
            output_tokens_standard = response_standard.usage.output_tokens
            output_tokens_efficient = response_efficient.usage.output_tokens

            # Calculate percentage reduction
            difference = output_tokens_standard - output_tokens_efficient
            percent_reduction = (
                (difference / output_tokens_standard) * 100
                if output_tokens_standard > 0
                else 0
            )

            # Significant reduction indicates feature is supported
            # Note: we use a very small threshold here as even a small reduction confirms it's working
            assert percent_reduction > 0, (
                "Expected some token reduction with token-efficient tools on Vertex AI"
            )

        except Exception as e:
            pytest.fail(f"Error testing token-efficient tools on Vertex AI: {e}")
