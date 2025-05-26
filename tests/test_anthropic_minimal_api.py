"""Minimal API test for Claude using the standardized approach.

This test demonstrates the standardized testing approach:
- Function-based tests for clarity
- Uses fixtures directly from conftest.py (not from conftest_api.py)
- Clear Arrange-Act-Assert structure
- Standard timing checks with timeouts
- Proper API fixture setup with minimal test environment dependencies
"""

import os
import time

import pytest
from llmproc.program import LLMProgram

# Minimal Claude model constant
CLAUDE_SMALL_MODEL = "claude-3-5-haiku-20241022"


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("Missing ANTHROPIC_API_KEY environment variable")
    return api_key


@pytest.fixture
def minimal_claude_program(anthropic_api_key):
    """Create a minimal Claude program for testing."""
    return LLMProgram(
        model_name=CLAUDE_SMALL_MODEL,
        provider="anthropic",
        system_prompt="You are a helpful assistant. Answer briefly.",
        parameters={"max_tokens": 100},
    )


@pytest.fixture
async def minimal_claude_process(minimal_claude_program):
    """Create a minimal Claude process for testing."""
    process = await minimal_claude_program.start()
    yield process


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Missing ANTHROPIC_API_KEY environment variable",
)
@pytest.mark.llm_api
@pytest.mark.anthropic_api
@pytest.mark.essential_api
@pytest.mark.asyncio
async def test_basic_response(minimal_claude_process):
    """Test that Claude responds with a basic answer.

    This test verifies the most basic functionality of the API,
    sending a simple arithmetic question and checking the response.
    """
    # Arrange
    process = minimal_claude_process
    prompt = "What is 2+2?"
    start_time = time.time()

    # Act
    run_result = await process.run(prompt)
    response = process.get_last_message()
    duration = time.time() - start_time

    # Assert
    assert duration < 8.0, f"Test took too long: {duration:.2f}s > 8.0s timeout"
    assert run_result is not None, "Run result should not be None"
    assert "4" in response, f"Expected '4' in response: {response}"


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Missing ANTHROPIC_API_KEY environment variable",
)
@pytest.mark.llm_api
@pytest.mark.anthropic_api
@pytest.mark.essential_api
@pytest.mark.asyncio
async def test_token_counting(minimal_claude_process):
    """Test that token counting works properly.

    This test verifies that the token counting functionality works
    with the Claude API, which is essential for monitoring usage.
    """
    # Arrange
    process = minimal_claude_process
    prompt = "Hello, how are you?"
    start_time = time.time()

    # Act
    await process.run(prompt)
    token_dict = await process.count_tokens()
    duration = time.time() - start_time

    # Assert
    assert duration < 8.0, f"Test took too long: {duration:.2f}s > 8.0s timeout"
    assert token_dict is not None, "Token count should not be None"
    assert isinstance(token_dict, dict), "Token count should be a dictionary"

    # Check flexible key pattern based on implementation
    if "total" in token_dict:
        assert token_dict["total"] > 0, "Total tokens should be greater than 0"
    elif "prompt" in token_dict:
        assert token_dict["prompt"] > 0, "Prompt tokens should be greater than 0"
    elif "input_tokens" in token_dict:
        assert token_dict["input_tokens"] > 0, "Input tokens should be greater than 0"
    else:
        raise AssertionError(f"Expected token count keys in token dict: {token_dict}")
