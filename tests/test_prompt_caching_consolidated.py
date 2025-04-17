"""Tests for the prompt caching functionality.

This file contains both unit tests for prompt caching functions and
integration tests for prompt caching using real API calls.
"""

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc import LLMProgram
from llmproc.common.results import RunResult
from llmproc.providers.anthropic_process_executor import (
    AnthropicProcessExecutor,
    add_cache_to_message,
    is_cacheable_content,
    state_to_api_messages,
    system_to_api_format,
    tools_to_api_format,
)
from tests.conftest_api import (
    claude_process_with_caching,
    claude_process_without_caching,
)

# =============================================================================
# UNIT TESTS - Test the prompt caching utility functions
# =============================================================================


def test_state_to_api_messages_no_cache():
    """Test state_to_api_messages with caching disabled."""
    # Arrange
    state = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    # Act
    result = state_to_api_messages(state, add_cache=False)

    # Assert
    # For the no-cache case, we expect a different object (deep copy) but same content
    assert id(state) != id(result)

    # Verify no cache_control was added
    for msg in result:
        if isinstance(msg.get("content"), list):
            for content in msg["content"]:
                assert "cache_control" not in content
        else:
            assert not isinstance(msg.get("content"), list) or "cache_control" not in msg["content"][0]


def has_cache_control(messages):
    """Helper function to check if any message has cache_control."""
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for content in msg["content"]:
                if isinstance(content, dict) and "cache_control" in content:
                    return True
    return False


def test_state_to_api_messages_with_cache():
    """Test state_to_api_messages with caching enabled."""
    # Arrange
    state = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    # Act
    result = state_to_api_messages(state, add_cache=True)

    # Assert
    # Verify the original state is not modified
    assert state[0]["content"] == "Hello"
    assert state[1]["content"] == "Hi there!"
    assert state[2]["content"] == "How are you?"

    # Verify cache_control was added to at least one message
    assert has_cache_control(result), "No cache_control found in any message"


def test_state_to_api_messages_with_tool_messages():
    """Test state_to_api_messages with tool messages."""
    # Arrange
    state = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Let me check something"},
        {
            "role": "tool",
            "tool_name": "calculator",
            "tool_args": {"a": 1, "b": 2},
            "content": "3",
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "content": "Result: 3"}],
        },
        {"role": "assistant", "content": "The result is 3"},
    ]

    # Act
    result = state_to_api_messages(state, add_cache=True)

    # Assert
    # Verify cache_control was added appropriately
    assert has_cache_control(result), "No cache_control found in any message"

    # Verify the cache control format
    cache_found = False
    for msg in result:
        if isinstance(msg.get("content"), list):
            for content in msg["content"]:
                if isinstance(content, dict) and "cache_control" in content:
                    assert content["cache_control"] == {"type": "ephemeral"}
                    cache_found = True

    assert cache_found, "No properly formatted cache_control found"


def test_system_to_api_format():
    """Test system_to_api_format function."""
    # Arrange
    system_prompt = "You are a helpful assistant."

    # Act - with caching enabled
    result_with_cache = system_to_api_format(system_prompt, add_cache=True)

    # Assert
    assert isinstance(result_with_cache, list)
    assert len(result_with_cache) == 1
    assert result_with_cache[0]["type"] == "text"
    assert result_with_cache[0]["text"] == system_prompt
    assert result_with_cache[0]["cache_control"] == {"type": "ephemeral"}

    # Act - with caching disabled
    result_without_cache = system_to_api_format(system_prompt, add_cache=False)

    # Assert
    assert result_without_cache == system_prompt  # No change should happen

    # Act - with empty system prompt
    empty_prompt = ""
    result_empty = system_to_api_format(empty_prompt, add_cache=True)

    # Assert
    assert result_empty == empty_prompt  # Should not add cache to empty content


def test_tools_to_api_format():
    """Test tools_to_api_format function."""
    # Arrange
    tools = [
        {"name": "tool1", "description": "Tool 1"},
        {"name": "tool2", "description": "Tool 2"},
    ]

    # Act - with caching enabled
    result_with_cache = tools_to_api_format(tools, add_cache=True)

    # Assert
    assert isinstance(result_with_cache, list)
    assert len(result_with_cache) == 2
    # Only the last tool should have cache_control
    assert "cache_control" not in result_with_cache[0]
    assert "cache_control" in result_with_cache[1]
    assert result_with_cache[1]["cache_control"] == {"type": "ephemeral"}

    # Act - with caching disabled
    result_without_cache = tools_to_api_format(tools, add_cache=False)

    # Assert
    assert isinstance(result_without_cache, list)
    assert len(result_without_cache) == 2
    # No tools should have cache_control
    assert "cache_control" not in result_without_cache[0]
    assert "cache_control" not in result_without_cache[1]

    # Act - with empty tools list
    empty_tools = []
    result_empty = tools_to_api_format(empty_tools, add_cache=True)

    # Assert
    assert result_empty == empty_tools  # Should not modify empty list


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


def test_add_cache_to_message():
    """Test add_cache_to_message function."""
    # Arrange - string content
    message = {"role": "user", "content": "Hello"}

    # Act
    add_cache_to_message(message)

    # Assert
    assert isinstance(message["content"], list)
    assert message["content"][0]["type"] == "text"
    assert message["content"][0]["text"] == "Hello"
    assert message["content"][0]["cache_control"] == {"type": "ephemeral"}

    # Arrange - list content
    message = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}

    # Act
    add_cache_to_message(message)

    # Assert
    assert message["content"][0]["cache_control"] == {"type": "ephemeral"}

    # Arrange - empty content
    message = {"role": "user", "content": ""}

    # Act
    add_cache_to_message(message)

    # Assert
    assert message["content"] == ""  # Should remain unchanged


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
