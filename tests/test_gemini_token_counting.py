"""Tests for Gemini token counting functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest

from llmproc.program import LLMProgram
from llmproc.providers.constants import PROVIDER_GEMINI, PROVIDER_GEMINI_VERTEX


@pytest.mark.llm_api
@pytest.mark.parametrize(
    "provider,model_name",
    [
        (PROVIDER_GEMINI, "gemini-2.0-flash"),
    ],
)
async def test_gemini_token_counting_api(provider, model_name):
    """Test the token counting API with real credentials."""
    # Skip if no API key
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY environment variable required")

    # Create program with system prompt
    program = LLMProgram(
        model_name=model_name,
        provider=provider,
        system_prompt="You are a helpful assistant. Be concise.",
        parameters={"temperature": 0.7},
    )

    # Start the process
    process = await program.start()

    # Get initial token count (just system prompt)
    initial_tokens = await process.count_tokens()
    assert "input_tokens" in initial_tokens
    assert "context_window" in initial_tokens
    assert "percentage" in initial_tokens
    assert "remaining_tokens" in initial_tokens
    assert "cached_tokens" in initial_tokens
    assert initial_tokens["input_tokens"] > 0
    assert initial_tokens["context_window"] > 0

    # Add a message and check token count increases
    await process.run("Hello, how are you?")
    after_message_tokens = await process.count_tokens()
    assert after_message_tokens["input_tokens"] > initial_tokens["input_tokens"]

    # Add a longer message and verify token count increases further
    await process.run(
        "Can you explain how token counting works in Gemini models? I want to understand the mechanism."
    )
    final_tokens = await process.count_tokens()
    assert final_tokens["input_tokens"] > after_message_tokens["input_tokens"]


async def test_gemini_token_counting_api_parsing():
    """Test token count parsing from API response."""
    # Import here to avoid mocking issues
    from llmproc.providers.gemini_process_executor import GeminiProcessExecutor

    # Create a mock process with minimal requirements
    class MockProcess:
        def __init__(self):
            self.model_name = "gemini-2.0-flash"
            self.state = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            self.enriched_system_prompt = "You are a helpful assistant."

            # Create mock client with count_tokens method
            self.client = MagicMock()
            self.client.models = MagicMock()

            # Mock token count response
            mock_token_response = MagicMock()
            mock_token_response.total_tokens = this_will_be_the_token_count = 42
            mock_token_response.cached_content_token_count = (
                this_will_be_cached_count
            ) = 10

            # Set the return value for count_tokens
            self.client.models.count_tokens.return_value = mock_token_response

    # Create executor and process instances
    executor = GeminiProcessExecutor()
    mock_process = MockProcess()

    # Test token counting with mocked API response
    token_info = await executor.count_tokens(mock_process)

    # Verify the token count information
    assert token_info["input_tokens"] == 42
    assert token_info["cached_tokens"] == 10
    assert "context_window" in token_info
    assert token_info["context_window"] > 0
    assert "percentage" in token_info
    assert "remaining_tokens" in token_info

    # Verify the count_tokens method was called with the correct model
    mock_process.client.models.count_tokens.assert_called_once()
    args, kwargs = mock_process.client.models.count_tokens.call_args
    assert kwargs.get("model") == "gemini-2.0-flash"


async def test_gemini_token_counting_fallback():
    """Test token counting fallback for unsupported clients directly."""
    # Import here to avoid having to mock the entire provider chain
    from llmproc.providers.gemini_process_executor import GeminiProcessExecutor

    # Create a mock process with minimal requirements
    class MockProcess:
        def __init__(self):
            self.model_name = "gemini-2.0-flash"
            self.state = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            self.enriched_system_prompt = "You are a helpful assistant."
            self.client = None  # Deliberately None to test fallback

    # Create executor and process instances
    executor = GeminiProcessExecutor()
    mock_process = MockProcess()

    # Test token counting with fallback mode
    token_info = await executor.count_tokens(mock_process)

    # Verify the token count information includes fallback indicators
    assert token_info["input_tokens"] == -1  # Indicates estimation
    assert "note" in token_info
    assert "not supported" in token_info["note"]
    assert "context_window" in token_info
    assert token_info["context_window"] > 0
    assert "percentage" in token_info
    assert "remaining_tokens" in token_info
