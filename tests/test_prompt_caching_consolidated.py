"""Tests for the prompt caching functionality.

This file contains both unit tests for prompt caching functions and
integration tests for prompt caching using real API calls.
"""

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llmproc import LLMProgram
from llmproc.common.constants import LLMPROC_MSG_ID
from llmproc.common.results import RunResult
from llmproc.providers.anthropic_utils import (
    apply_cache_control,
    format_state_to_api_messages,
    format_system_prompt,
    is_cacheable_content,
    prepare_api_request,
)

from tests.conftest_api import (
    claude_process_with_caching,
    claude_process_without_caching,
)

# =============================================================================
# UNIT TESTS - Test the prompt caching utility functions
# =============================================================================


def has_cache_control(messages):
    """Helper function to check if any message has cache_control."""
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for content in msg["content"]:
                if isinstance(content, dict) and "cache_control" in content:
                    return True
    return False


def test_run_result_cache_metrics():
    """Test cache metrics in RunResult."""
    # Arrange
    run_result = RunResult()

    # Act
    run_result.add_api_call(
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 80,
            }
        }
    )

    # Assert
    assert run_result.input_tokens == 100
    assert run_result.output_tokens == 50
    assert run_result.cached_tokens == 20
    assert run_result.cache_write_tokens == 80


def test_is_cacheable_content():
    """Test is_cacheable_content function."""
    # Arrange - Empty content cases
    empty_cases = [
        "",  # Empty string
        "   ",  # Whitespace only
        None,  # None value
        {"type": "text", "text": ""},  # Empty text in dict
        {"type": "tool_result", "content": ""},  # Empty content in dict
    ]

    # Arrange - Valid content cases
    valid_cases = [
        "Hello",  # Simple string
        {"type": "text", "text": "Hello"},  # Dict with text
        {"type": "tool_result", "content": "Result"},  # Dict with content
    ]

    # Act & Assert - Empty content should not be cacheable
    for empty_case in empty_cases:
        assert not is_cacheable_content(empty_case), f"Empty case should not be cacheable: {empty_case}"

    # Act & Assert - Valid content should be cacheable
    for valid_case in valid_cases:
        assert is_cacheable_content(valid_case), f"Valid case should be cacheable: {valid_case}"


# =============================================================================
# INTEGRATION TESTS - Test prompt caching with real API calls
# =============================================================================


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.asyncio
async def test_basic_caching():
    """Test that prompt caching works with minimal API calls."""
    # Skip if no API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")

    # Arrange - create a large system prompt that exceeds min cacheable size (1024 tokens)
    long_system_prompt = "You are a helpful assistant. " + ("This is placeholder content. " * 1000)

    # Arrange - create program with automatic caching enabled (default)
    program = LLMProgram(
        model_name="claude-3-5-sonnet-20240620",
        provider="anthropic",
        system_prompt=long_system_prompt,
        parameters={"max_tokens": 100},
    )

    # Act - create process and make two calls to trigger caching
    process = await program.start()
    result1 = await process.run("Hello, who are you?")
    result2 = await process.run("Tell me a joke.")

    # Assert - at least one call should show cache activity
    total_cache_activity = (
        result1.cached_tokens + result1.cache_write_tokens + result2.cached_tokens + result2.cache_write_tokens
    )
    assert total_cache_activity > 0, "Should have some cache activity"


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.asyncio
async def test_caching_integration(claude_process_with_caching):
    """Test prompt caching with a real API call."""
    # Arrange
    process = claude_process_with_caching
    start_time = time.time()

    # Act - two API calls, second should use cache
    result1 = await process.run("Tell me a short story")
    result2 = await process.run("Tell me another short story")
    duration = time.time() - start_time

    # Assert - verify API calls occurred
    assert result1.api_calls > 0, "No API calls recorded in first result"
    assert result2.api_calls > 0, "No API calls recorded in second result"

    # Assert - verify responses are different
    state = process.get_state()
    assert len(state) >= 4, "Expected at least 4 messages in state"
    assert state[-2]["content"] != state[-4]["content"], "Response messages should be different"

    # Timing assertion (not strictly necessary but good practice)
    assert duration < 60.0, f"Test took too long: {duration:.2f} seconds"


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.asyncio
async def test_multi_turn_caching(claude_process_with_caching):
    """Test caching with a multi-turn conversation."""
    # Arrange
    process = claude_process_with_caching
    start_time = time.time()
    turns = [
        "Hello, how are you?",
        "What's your favorite color?",
        "Why do you like that color?",
    ]

    # Act - run multiple turns
    results = []
    for turn in turns:
        result = await process.run(turn)
        results.append(result)

    duration = time.time() - start_time

    # Assert - verify we got responses for all turns
    assert len(results) == len(turns), f"Expected {len(turns)} results, got {len(results)}"

    # Assert - verify we have conversation history
    state = process.get_state()
    assert len(state) > len(turns), "State should contain system prompt plus all turns"

    # Timing assertion (not strictly necessary but good practice)
    assert duration < 60.0, f"Test took too long: {duration:.2f} seconds"


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.asyncio
async def test_disable_automatic_caching(claude_process_with_caching, claude_process_without_caching):
    """Test disabling automatic caching."""
    # Arrange
    process_with_caching_disabled = claude_process_without_caching
    process_with_caching_enabled = claude_process_with_caching
    start_time = time.time()

    # Act - make API calls with both processes
    result_disabled = await process_with_caching_disabled.run("Hello, how are you?")
    result_enabled = await process_with_caching_enabled.run("Hello, how are you?")
    duration = time.time() - start_time

    # Assert - both processes should have API calls
    assert result_disabled.api_calls > 0, "No API calls recorded with caching disabled"
    assert result_enabled.api_calls > 0, "No API calls recorded with caching enabled"

    # Assert - both processes should produce valid responses
    assert process_with_caching_disabled.get_last_message(), "No response from process with caching disabled"
    assert process_with_caching_enabled.get_last_message(), "No response from process with caching enabled"

    # Timing assertion (not strictly necessary but good practice)
    assert duration < 60.0, f"Test took too long: {duration:.2f} seconds"


def test_format_state_to_api_messages():
    """Test format_state_to_api_messages function."""
    # Arrange
    state = [
        {"role": "user", "content": "Hello", LLMPROC_MSG_ID: 1},
        {"role": "assistant", "content": "Hi there!", LLMPROC_MSG_ID: 2},
        {"role": "user", "content": [{"type": "text", "text": "How are you?"}], LLMPROC_MSG_ID: 3},
    ]

    # Act
    from llmproc.providers.anthropic_utils import format_state_to_api_messages

    result = format_state_to_api_messages(state)

    # Assert
    # Verify message IDs are added correctly
    assert len(result) == 3
    assert LLMPROC_MSG_ID not in result[0], "LLMPROC_MSG_ID should be removed"
    assert LLMPROC_MSG_ID not in result[1], "LLMPROC_MSG_ID should be removed"
    assert LLMPROC_MSG_ID not in result[2], "LLMPROC_MSG_ID should be removed"

    # Check first message (string content)
    assert isinstance(result[0]["content"], list)
    assert result[0]["content"][0]["type"] == "text"
    assert "[msg_1]" in result[0]["content"][0]["text"]

    # Check second message (string content)
    assert isinstance(result[1]["content"], list)
    assert result[1]["content"][0]["type"] == "text"
    assert "[msg_2]" not in result[1]["content"][0]["text"]

    # Check third message (list content)
    assert isinstance(result[2]["content"], list)
    assert result[2]["content"][0]["type"] == "text"
    assert "[msg_3]" in result[2]["content"][0]["text"]


def test_format_system_prompt():
    """Test format_system_prompt function."""
    # Arrange
    string_prompt = "You are a helpful assistant."
    list_prompt = [{"type": "text", "text": "You are a helpful assistant."}]

    # Act
    from llmproc.providers.anthropic_utils import format_system_prompt

    string_result = format_system_prompt(string_prompt)
    list_result = format_system_prompt(list_prompt)

    # Assert
    # String prompt should be converted to list format
    assert isinstance(string_result, list)
    assert len(string_result) == 1
    assert string_result[0]["type"] == "text"
    assert string_result[0]["text"] == string_prompt

    # List prompt should remain in list format
    assert isinstance(list_result, list)
    assert len(list_result) == 1
    assert list_result[0]["type"] == "text"
    assert list_result[0]["text"] == "You are a helpful assistant."

    # Ensure no cache control was added
    assert "cache_control" not in string_result[0]
    assert "cache_control" not in list_result[0]


def test_apply_cache_control():
    """Test apply_cache_control function."""
    # Arrange
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
    ]
    system = [{"type": "text", "text": "You are a helpful assistant."}]
    tools = [{"name": "calculator", "description": "A calculator tool"}]

    # Act
    from llmproc.providers.anthropic_utils import apply_cache_control

    cached_messages, cached_system, cached_tools = apply_cache_control(messages, system, tools)

    # Assert
    # Verify system prompt has cache control
    assert "cache_control" in cached_system[0]
    assert cached_system[0]["cache_control"] == {"type": "ephemeral"}

    # Verify last 3 messages have cache control
    assert "cache_control" in cached_messages[0]["content"][0]
    assert "cache_control" in cached_messages[1]["content"][0]
    assert "cache_control" in cached_messages[2]["content"][0]

    # Verify tools don't have cache control
    assert "cache_control" not in cached_tools[0]

    # Verify original messages/system are not modified
    assert "cache_control" not in messages[0]["content"][0]
    assert "cache_control" not in system[0]


def test_prepare_api_request():
    """Test prepare_api_request function."""
    # Arrange
    from unittest.mock import MagicMock

    # Create a mock process
    process = MagicMock()
    process.state = [
        {"role": "user", "content": "Hello", LLMPROC_MSG_ID: 1},
        {"role": "assistant", "content": "Hi there!", LLMPROC_MSG_ID: 2},
    ]
    process.enriched_system_prompt = "You are a helpful assistant."
    process.tools = [{"name": "calculator", "description": "A calculator tool"}]
    process.model_name = "claude-3-7-sonnet"
    process.api_params = {"max_tokens": 1000}
    process.disable_automatic_caching = False
    process.provider = "anthropic"
    process.tool_manager.message_ids_enabled = True

    # Act
    from llmproc.providers.anthropic_utils import prepare_api_request

    api_request = prepare_api_request(process)

    # Assert
    # Verify structure of API request
    assert "model" in api_request
    assert "system" in api_request
    assert "messages" in api_request
    assert "tools" in api_request
    assert "max_tokens" in api_request

    # Check model name
    assert api_request["model"] == "claude-3-7-sonnet"

    # Check max_tokens
    assert api_request["max_tokens"] == 1000

    # System prompt is now a string, not a list with cache_control

    # Verify messages have message IDs and cache control
    assert "[msg_1]" in api_request["messages"][0]["content"][0]["text"]
    assert "[msg_2]" not in api_request["messages"][1]["content"][0]["text"]
    assert "cache_control" in api_request["messages"][0]["content"][0]
    assert "cache_control" in api_request["messages"][1]["content"][0]

    # Test with caching disabled
    api_request_no_cache = prepare_api_request(process, add_cache=False)
    assert "cache_control" not in api_request_no_cache["system"][0]
    assert "cache_control" not in api_request_no_cache["messages"][0]["content"][0]
