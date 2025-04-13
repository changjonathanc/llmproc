"""Tests for the prompt caching implementation."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.common.results import RunResult
from llmproc.providers.anthropic_process_executor import (
    AnthropicProcessExecutor,
    add_cache_to_message,
    is_cacheable_content,
    state_to_api_messages,
    system_to_api_format,
    tools_to_api_format,
)


class TestPromptCaching:
    """Test suite for the prompt caching functionality."""

    def test_state_to_api_messages_no_cache(self):
        """Test state_to_api_messages with caching disabled."""
        state = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        # Call the function with add_cache=False
        result = state_to_api_messages(state, add_cache=False)

        # For the no-cache case, we expect a different object (deep copy) but same content
        assert id(state) != id(result)

        # Verify no cache_control was added
        for msg in result:
            if isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    assert "cache_control" not in content
            else:
                assert (
                    not isinstance(msg.get("content"), list)
                    or "cache_control" not in msg["content"][0]
                )

    def test_state_to_api_messages_with_cache(self):
        """Test state_to_api_messages with caching enabled."""
        state = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        # Call the function with add_cache=True
        result = state_to_api_messages(state, add_cache=True)

        # Verify the original state is not modified
        assert state[0]["content"] == "Hello"
        assert state[1]["content"] == "Hi there!"
        assert state[2]["content"] == "How are you?"

        # Verify cache_control was added to at least one message
        cache_added = False
        for msg in result:
            if isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    if isinstance(content, dict) and "cache_control" in content:
                        cache_added = True
                        break
            if cache_added:
                break

        assert cache_added, "No cache_control found in any message"

    def test_state_to_api_messages_with_tool_messages(self):
        """Test state_to_api_messages with tool messages."""
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

        # Call the function with add_cache=True
        result = state_to_api_messages(state, add_cache=True)

        # Verify cache_control was added appropriately
        # The last message should have cache control
        for msg in result:
            if isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    if isinstance(content, dict) and "cache_control" in content:
                        # Confirm it's the right format
                        assert content["cache_control"] == {"type": "ephemeral"}

    def test_system_to_api_format(self):
        """Test system_to_api_format function."""
        system_prompt = "You are a helpful assistant."

        # Test with caching enabled
        result_with_cache = system_to_api_format(system_prompt, add_cache=True)
        assert isinstance(result_with_cache, list)
        assert len(result_with_cache) == 1
        assert result_with_cache[0]["type"] == "text"
        assert result_with_cache[0]["text"] == system_prompt
        assert result_with_cache[0]["cache_control"] == {"type": "ephemeral"}

        # Test with caching disabled
        result_without_cache = system_to_api_format(system_prompt, add_cache=False)
        assert result_without_cache == system_prompt  # No change should happen

        # Test with empty system prompt
        empty_prompt = ""
        result_empty = system_to_api_format(empty_prompt, add_cache=True)
        assert result_empty == empty_prompt  # Should not add cache to empty content

    def test_tools_to_api_format(self):
        """Test tools_to_api_format function."""
        tools = [
            {"name": "tool1", "description": "Tool 1"},
            {"name": "tool2", "description": "Tool 2"},
        ]

        # Test with caching enabled
        result_with_cache = tools_to_api_format(tools, add_cache=True)
        assert isinstance(result_with_cache, list)
        assert len(result_with_cache) == 2
        # Only the last tool should have cache_control
        assert "cache_control" not in result_with_cache[0]
        assert "cache_control" in result_with_cache[1]
        assert result_with_cache[1]["cache_control"] == {"type": "ephemeral"}

        # Test with caching disabled
        result_without_cache = tools_to_api_format(tools, add_cache=False)
        assert isinstance(result_without_cache, list)
        assert len(result_without_cache) == 2
        # No tools should have cache_control
        assert "cache_control" not in result_without_cache[0]
        assert "cache_control" not in result_without_cache[1]

        # Test with empty tools list
        empty_tools = []
        result_empty = tools_to_api_format(empty_tools, add_cache=True)
        assert result_empty == empty_tools  # Should not modify empty list

    def test_run_result_cache_metrics(self):
        """Test cache metrics in RunResult."""
        run_result = RunResult()

        # Add an API call with caching metrics
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

        # Test metrics
        assert run_result.input_tokens == 100
        assert run_result.output_tokens == 50
        assert run_result.cached_tokens == 20
        assert run_result.cache_write_tokens == 80

    def test_is_cacheable_content(self):
        """Test is_cacheable_content function."""
        # Test with empty content
        assert not is_cacheable_content("")
        assert not is_cacheable_content("   ")
        assert not is_cacheable_content(None)

        # Test with valid content
        assert is_cacheable_content("Hello")
        assert is_cacheable_content({"type": "text", "text": "Hello"})
        assert is_cacheable_content({"type": "tool_result", "content": "Result"})

        # Test with invalid content
        assert not is_cacheable_content({"type": "text", "text": ""})
        assert not is_cacheable_content({"type": "tool_result", "content": ""})

    def test_add_cache_to_message(self):
        """Test add_cache_to_message function."""
        # Test with string content
        message = {"role": "user", "content": "Hello"}
        add_cache_to_message(message)
        assert isinstance(message["content"], list)
        assert message["content"][0]["type"] == "text"
        assert message["content"][0]["text"] == "Hello"
        assert message["content"][0]["cache_control"] == {"type": "ephemeral"}

        # Test with list content
        message = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        add_cache_to_message(message)
        assert message["content"][0]["cache_control"] == {"type": "ephemeral"}

        # Test with empty content (should not add cache)
        message = {"role": "user", "content": ""}
        add_cache_to_message(message)
        assert message["content"] == ""  # Should remain unchanged
