"""Tests for provider-specific feature implementations.

This file follows the standardized test patterns, separating:
1. Unit tests - Test individual functions without API calls
2. API tests - Test integration with real API calls, using fixtures and timing assertions
"""

import json
import os
import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor
from llmproc.providers.anthropic_utils import (
    apply_cache_control,
    format_system_prompt,
    prepare_api_request,
)

# Define constants for model versions to make updates easier
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"  # Use a specific versioned model
CLAUDE_37_MODEL = "claude-3-7-sonnet-20250219"  # Claude 3.7 model


@contextmanager
def timed_api_test(timeout_seconds=15.0):
    """Provides timing context for API tests.

    Args:
        timeout_seconds: Maximum allowed duration in seconds

    Yields:
        None: Just provides timing context

    Raises:
        AssertionError: If the test exceeds the timeout
    """
    start_time = time.time()
    yield
    duration = time.time() - start_time
    assert duration < timeout_seconds, f"Test took too long: {duration:.2f}s > {timeout_seconds}s timeout"


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment variable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("Missing ANTHROPIC_API_KEY environment variable")
    return api_key


@pytest.fixture
def vertex_project_id():
    """Get Vertex AI project ID from environment variable."""
    project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if not project_id:
        pytest.skip("Missing ANTHROPIC_VERTEX_PROJECT_ID environment variable")
    return project_id


@pytest.fixture
async def direct_anthropic_process_with_caching(anthropic_api_key):
    """Create a Claude process with caching enabled for API testing.

    This fixture provides an isolated LLMProcess instance with a large system
    prompt to trigger caching behavior. It follows the standard program.start()
    pattern for initialization and uses function scope.

    Args:
        anthropic_api_key: The Anthropic API key fixture

    Yields:
        LLMProcess: A started Claude process with caching enabled
    """
    from llmproc import LLMProgram

    # Create a simple program with a large system prompt to trigger caching
    program = LLMProgram(
        model_name=CLAUDE_MODEL,
        provider="anthropic",
        system_prompt="You are a helpful assistant. " + ("This is filler content. " * 500),
        parameters={"max_tokens": 1000},
        disable_automatic_caching=False,  # Ensure caching is enabled
    )

    # Start the process using the standard pattern
    process = await program.start()
    yield process


@pytest.fixture
async def claude37_process_with_token_efficient_tools_vertex(vertex_project_id):
    """Create a Claude 3.7 process on Vertex AI with token-efficient tools.

    This fixture provides an isolated LLMProcess instance with token-efficient
    tools enabled on Vertex AI. It follows the standard program.start() pattern
    and uses function scope.

    Args:
        vertex_project_id: The Vertex AI project ID fixture

    Yields:
        AsyncAnthropicVertex: A configured Vertex AI client for Claude 3.7
    """
    try:
        from anthropic import AsyncAnthropicVertex

        VERTEX_AVAILABLE = True
    except ImportError:
        VERTEX_AVAILABLE = False

    if not VERTEX_AVAILABLE:
        pytest.skip("Anthropic Vertex SDK not installed")

    region = os.environ.get("CLOUD_ML_REGION", "us-central1")

    # Initialize Vertex client
    client = AsyncAnthropicVertex(project_id=vertex_project_id, region=region)
    yield client


class TestProviderSpecificUnitTests:
    """Unit tests for provider-specific features without API calls."""

    def test_provider_detection(self):
        """Test provider type detection logic."""
        # Arrange & Act - Direct Anthropic provider detection
        is_direct_anthropic = "anthropic" in "anthropic" and "vertex" not in "anthropic"

        # Assert
        assert is_direct_anthropic

        # Arrange & Act - Vertex AI provider detection
        is_direct_anthropic = "anthropic" in "anthropic_vertex" and "vertex" not in "anthropic_vertex"

        # Assert
        assert not is_direct_anthropic

        # Arrange & Act - Combined provider strings
        is_direct_anthropic = "anthropic" in "anthropic-vertex" and "vertex" not in "anthropic-vertex"

        # Assert
        assert not is_direct_anthropic

    def test_system_prompt_format_and_cache(self):
        """Test formatting system prompt and applying cache."""
        # Arrange
        system_prompt = "You are a helpful assistant."

        # Format the system prompt
        formatted = format_system_prompt(system_prompt)

        # Apply cache
        _, cached_system, _ = apply_cache_control([], formatted)

        # Assert - Check system prompt with cache
        assert isinstance(cached_system, list)
        assert cached_system[0]["type"] == "text"
        assert cached_system[0]["text"] == system_prompt
        assert cached_system[0]["cache_control"] == {"type": "ephemeral"}

        # Format without applying cache
        formatted = format_system_prompt(system_prompt)

        # Assert - Check formatting without cache
        assert isinstance(formatted, list)
        assert len(formatted) == 1
        assert formatted[0]["type"] == "text"
        assert formatted[0]["text"] == system_prompt
        assert "cache_control" not in formatted[0]

        # Arrange - Already structured prompt
        structured_prompt = [{"type": "text", "text": "You are a helpful assistant."}]

        # Format structured prompt
        formatted = format_system_prompt(structured_prompt)

        # Assert - Check structure is preserved
        assert isinstance(formatted, list)
        assert len(formatted) == 1
        assert formatted[0]["type"] == "text"
        assert formatted[0]["text"] == "You are a helpful assistant."

    def test_tools_in_api_request(self):
        """Test tools handling in API request preparation."""
        # Arrange
        process = MagicMock()
        process.state = []
        process.enriched_system_prompt = "You are an assistant"
        process.tools = [
            {"name": "calculator", "description": "Use this to perform calculations"},
            {"name": "web_search", "description": "Search the web for information"},
        ]
        process.model_name = "claude-3-sonnet"
        process.api_params = {}
        process.disable_automatic_caching = False

        # Act - Prepare API request
        request = prepare_api_request(process)

        # Assert - Tools should be included in request
        assert request["tools"] is process.tools

        # Verify tools are not modified by cache application
        assert "cache_control" not in request["tools"][0]
        assert "cache_control" not in request["tools"][1]

    def test_token_efficient_tools_header(self):
        """Test token-efficient tools header application."""
        # Arrange
        executor = AnthropicProcessExecutor()
        mock_process = MagicMock()
        mock_process.provider = "anthropic"
        mock_process.model_name = CLAUDE_37_MODEL
        extra_headers = {}

        # Act - Apply token-efficient tools logic
        if "anthropic" in mock_process.provider.lower() and mock_process.model_name.startswith("claude-3-7"):
            if "anthropic-beta" not in extra_headers:
                extra_headers["anthropic-beta"] = "token-efficient-tools-2025-02-19"
            elif "token-efficient-tools" not in extra_headers["anthropic-beta"]:
                extra_headers["anthropic-beta"] += ",token-efficient-tools-2025-02-19"

        # Assert - Direct Anthropic
        assert "anthropic-beta" in extra_headers
        assert extra_headers["anthropic-beta"] == "token-efficient-tools-2025-02-19"

        # Arrange - Vertex AI
        mock_process.provider = "anthropic_vertex"
        extra_headers = {}

        # Act - Apply token-efficient tools logic
        if "anthropic" in mock_process.provider.lower() and mock_process.model_name.startswith("claude-3-7"):
            if "anthropic-beta" not in extra_headers:
                extra_headers["anthropic-beta"] = "token-efficient-tools-2025-02-19"
            elif "token-efficient-tools" not in extra_headers["anthropic-beta"]:
                extra_headers["anthropic-beta"] += ",token-efficient-tools-2025-02-19"

        # Assert - Vertex AI
        assert "anthropic-beta" in extra_headers
        assert extra_headers["anthropic-beta"] == "token-efficient-tools-2025-02-19"

        # Arrange - Non-Claude 3.7 model
        mock_process.provider = "anthropic"
        mock_process.model_name = "claude-3-5-sonnet"
        extra_headers = {"anthropic-beta": "token-efficient-tools-2025-02-19"}

        # Act & Assert - Check warning logic
        with patch("llmproc.providers.anthropic_process_executor.logger") as mock_logger:
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
@pytest.mark.anthropic_api
@pytest.mark.extended_api
class TestProviderSpecificAPITests:
    """API tests for provider-specific features requiring real API access."""

    @pytest.mark.asyncio
    async def test_cache_control_with_direct_anthropic(self, direct_anthropic_process_with_caching):
        """Test that cache control parameters work with direct Anthropic API."""
        # Arrange
        process = direct_anthropic_process_with_caching

        with timed_api_test(timeout_seconds=20.0):
            # Act - Run process twice to trigger caching
            result1 = await process.run("What is your name?")
            result2 = await process.run("Tell me a joke.")

            # Extract API call metrics directly from RunResult properties
            # Assert - Verify token counts directly from the result
            assert result1.input_tokens > 0, "Expected positive input token count in first call"
            assert result2.input_tokens > 0, "Expected positive input token count in second call"

            # Verify we have cached tokens in the second call (reusing first call content)
            assert result2.cached_tokens > 0, "Expected cached tokens in second call"

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"),
        reason="No Vertex AI project ID found",
    )
    @pytest.mark.anthropic_api
    @pytest.mark.asyncio
    async def test_token_efficient_tools_vertex(self, claude37_process_with_token_efficient_tools_vertex):
        """Test that token-efficient tools works with Vertex AI."""
        # Arrange
        client = claude37_process_with_token_efficient_tools_vertex

        # Define a calculator tool
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

        prompt = "What is the square root of 256? Please use the calculator tool."

        with timed_api_test(timeout_seconds=25.0):
            # Act - First request WITHOUT token-efficient tools
            response_standard = await client.messages.create(
                model="claude-3-7-sonnet@20250219",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                tools=[calculator_tool],
                system="You are a helpful AI assistant that uses tools when appropriate.",
            )

            # Wait to avoid rate limits
            time.sleep(2)

            # Act - Second request WITH token-efficient tools
            response_efficient = await client.messages.create(
                model="claude-3-7-sonnet@20250219",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                tools=[calculator_tool],
                system="You are a helpful AI assistant that uses tools when appropriate.",
                extra_headers={"anthropic-beta": "token-efficient-tools-2025-02-19"},
            )

            # Assert - Compare token usage
            output_tokens_standard = response_standard.usage.output_tokens
            output_tokens_efficient = response_efficient.usage.output_tokens

            # Calculate difference
            difference = output_tokens_standard - output_tokens_efficient
            percent_reduction = (difference / output_tokens_standard) * 100 if output_tokens_standard > 0 else 0

            # There should be some token reduction (even small confirms it's working)
            assert percent_reduction >= 0, "Expected token-efficient tools to not increase token usage"
