"""Tests for the Anthropic utilities module."""

import copy
from unittest.mock import MagicMock, patch

import pytest

from llmproc.providers.anthropic_utils import (
    add_cache_to_message,
    add_message_ids,
    add_token_efficient_header_if_needed,
    contains_tool_calls,
    get_context_window_size,
    is_cacheable_content,
    safe_callback,
    state_to_api_messages,
    system_to_api_format,
    tools_to_api_format,
)
from llmproc.providers.constants import ANTHROPIC_PROVIDERS


class TestCacheControl:
    """Tests for the cache control functions."""

    def test_is_cacheable_content_empty(self):
        """Test that empty content is not cacheable."""
        assert is_cacheable_content(None) is False
        assert is_cacheable_content("") is False
        assert is_cacheable_content(" ") is False

    def test_is_cacheable_content_string(self):
        """Test that non-empty strings are cacheable."""
        assert is_cacheable_content("content") is True

    def test_is_cacheable_content_dict(self):
        """Test that dictionaries with content are cacheable."""
        assert is_cacheable_content({"type": "text", "text": "content"}) is True
        assert (
            is_cacheable_content({"type": "tool_result", "content": "result"}) is True
        )
        assert is_cacheable_content({"type": "text", "text": ""}) is False
        assert (
            is_cacheable_content({"type": "other"}) is True
        )  # Default is True for other types

    def test_add_cache_to_message_list_content(self):
        """Test adding cache to a message with list content."""
        message = {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}
        add_cache_to_message(message)
        assert message["content"][0].get("cache_control") == {"type": "ephemeral"}

    def test_add_cache_to_message_string_content(self):
        """Test adding cache to a message with string content."""
        message = {"role": "assistant", "content": "Hello"}
        add_cache_to_message(message)
        assert isinstance(message["content"], list)
        assert message["content"][0].get("cache_control") == {"type": "ephemeral"}
        assert message["content"][0].get("text") == "Hello"

    def test_add_cache_to_message_empty_content(self):
        """Test that cache is not added to empty content."""
        message = {"role": "assistant", "content": ""}
        message_copy = copy.deepcopy(message)
        add_cache_to_message(message)
        # Message should be unchanged
        assert message == message_copy

    def test_add_cache_to_message_only_adds_to_first_eligible(self):
        """Test that cache is only added to the first eligible content."""
        message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }
        add_cache_to_message(message)
        assert message["content"][0].get("cache_control") == {"type": "ephemeral"}
        assert message["content"][1].get("cache_control") is None


class TestMessageFormatting:
    """Tests for message formatting functions."""

    def test_add_message_ids_string_content(self):
        """Test adding message IDs to messages with string content."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = add_message_ids(messages)
        assert result[0]["content"].startswith("[msg_0]")
        assert result[1]["content"].startswith("[msg_1]")

    def test_add_message_ids_list_content(self):
        """Test adding message IDs to messages with list content."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]
        result = add_message_ids(messages)
        assert result[0]["content"][0]["text"].startswith("[msg_0]")
        assert result[1]["content"][0]["text"].startswith("[msg_1]")

    def test_add_message_ids_custom_goto_id(self):
        """Test that custom goto_ids are used and then removed."""
        messages = [
            {"role": "user", "content": "Hello", "goto_id": "custom_id"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = add_message_ids(messages)
        assert result[0]["content"].startswith("[custom_id]")
        assert "goto_id" not in result[0]
        assert result[1]["content"].startswith("[msg_1]")

    def test_state_to_api_messages_with_caching(self):
        """Test transforming state to API messages with caching."""
        # Start with the simplest case to verify caching behavior
        state = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = state_to_api_messages(state, add_cache=True)

        # Just verify that the last message has cache
        assert len(result) == 2
        last_message = result[-1]
        assert last_message["role"] == "assistant"
        assert isinstance(last_message["content"], list)
        assert last_message["content"][0].get("cache_control") == {"type": "ephemeral"}

        # Also verify that the message IDs are added
        assert "[msg_0]" in result[0]["content"]
        assert "[msg_1]" in last_message["content"][0]["text"]

    def test_state_to_api_messages_without_caching(self):
        """Test transforming state to API messages without caching."""
        state = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = state_to_api_messages(state, add_cache=False)

        # No message should have cache
        for msg in result:
            if isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    assert content.get("cache_control") is None


class TestAPIFormatting:
    """Tests for API formatting functions."""

    def test_system_to_api_format_string(self):
        """Test converting string system prompt to API format."""
        system = "Hello, I am Claude"
        result = system_to_api_format(system, add_cache=True)

        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[0]["text"] == system
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_system_to_api_format_empty(self):
        """Test that empty system prompts are not modified."""
        system = ""
        result = system_to_api_format(system, add_cache=True)
        assert result == system

    def test_system_to_api_format_no_cache(self):
        """Test system prompt formatting without caching."""
        system = "Hello, I am Claude"
        result = system_to_api_format(system, add_cache=False)
        assert result == system

    def test_tools_to_api_format_with_cache(self):
        """Test tools formatting with caching."""
        tools = [
            {"name": "calculator", "description": "A calculator tool"},
            {"name": "weather", "description": "A weather tool"},
        ]
        result = tools_to_api_format(tools, add_cache=True)

        # Last tool should have cache
        assert result[-1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[0]

    def test_tools_to_api_format_empty(self):
        """Test tools formatting with empty tools."""
        tools = []
        result = tools_to_api_format(tools, add_cache=True)
        assert result == tools

    def test_tools_to_api_format_none(self):
        """Test tools formatting with None."""
        result = tools_to_api_format(None, add_cache=True)
        assert result is None


class TestTokenEfficientHeaders:
    """Tests for token efficient headers functions."""

    def test_add_token_efficient_header_empty_headers(self):
        """Test adding token-efficient header to empty headers."""
        process = MagicMock()
        process.provider = "anthropic"
        process.model_name = "claude-3-7-sonnet-20250219"

        headers = {}
        result = add_token_efficient_header_if_needed(process, headers)

        assert "anthropic-beta" in result
        assert result["anthropic-beta"] == "token-efficient-tools-2025-02-19"

    def test_add_token_efficient_header_existing_headers(self):
        """Test adding token-efficient header to existing headers."""
        process = MagicMock()
        process.provider = "anthropic"
        process.model_name = "claude-3-7-sonnet-20250219"

        headers = {"anthropic-beta": "existing-feature"}
        result = add_token_efficient_header_if_needed(process, headers)

        assert "anthropic-beta" in result
        assert (
            "existing-feature,token-efficient-tools-2025-02-19"
            == result["anthropic-beta"]
        )

    def test_add_token_efficient_header_already_present(self):
        """Test not duplicating token-efficient header if already present."""
        process = MagicMock()
        process.provider = "anthropic"
        process.model_name = "claude-3-7-sonnet-20250219"

        headers = {"anthropic-beta": "token-efficient-tools-2025-02-19"}
        result = add_token_efficient_header_if_needed(process, headers)

        assert "anthropic-beta" in result
        assert result["anthropic-beta"] == "token-efficient-tools-2025-02-19"

    def test_add_token_efficient_header_non_claude_37(self):
        """Test that header is not added for non-Claude 3.7 models."""
        process = MagicMock()
        process.provider = "anthropic"
        process.model_name = "claude-3-5-sonnet-20241022"

        headers = {}
        result = add_token_efficient_header_if_needed(process, headers)

        assert not headers  # Original headers unchanged
        assert result == headers  # Result headers should be empty


class TestSafeCallback:
    """Tests for the safe callback function."""

    def test_safe_callback_successful_execution(self):
        """Test successful callback execution."""
        callback_fn = MagicMock()
        safe_callback(callback_fn, "arg1", "arg2", callback_name="test_callback")

        callback_fn.assert_called_once_with("arg1", "arg2")

    @patch("llmproc.providers.anthropic_utils.logger")
    def test_safe_callback_handles_exception(self, mock_logger):
        """Test that exceptions in callbacks are caught and logged."""
        callback_fn = MagicMock(side_effect=Exception("Test error"))

        # This should not raise an exception
        safe_callback(callback_fn, "arg1", callback_name="test_callback")

        callback_fn.assert_called_once_with("arg1")
        mock_logger.warning.assert_called_once()
        assert "Error in test_callback callback" in mock_logger.warning.call_args[0][0]

    def test_safe_callback_none_callback(self):
        """Test handling None callback."""
        # Should not raise an exception
        safe_callback(None, "arg1", callback_name="test_callback")


class TestMiscUtils:
    """Tests for miscellaneous utility functions."""

    def test_contains_tool_calls_with_tool_use(self):
        """Test detecting tool calls in response content."""
        # Create mock content with a tool_use item
        content = [
            MagicMock(type="text", text="Some text"),
            MagicMock(type="tool_use", name="test_tool", input={"arg": "value"}),
        ]

        assert contains_tool_calls(content) is True

    def test_contains_tool_calls_without_tool_use(self):
        """Test detecting no tool calls in response content."""
        # Create mock content with only text items
        content = [
            MagicMock(type="text", text="Some text"),
            MagicMock(type="text", text="More text"),
        ]

        assert contains_tool_calls(content) is False

    def test_contains_tool_calls_with_malformed_content(self):
        """Test handling content items without type attribute."""
        # Create mock content with items missing type attribute
        content = [
            MagicMock(spec=["text"]),  # No type attribute
            MagicMock(),  # Empty mock
        ]

        assert contains_tool_calls(content) is False

    def test_get_context_window_size(self):
        """Test getting context window size for various models."""
        window_sizes = {
            "claude-3-5": 200000,
            "claude-3-7": 250000,
        }

        # Test exact match
        assert get_context_window_size("claude-3-5-sonnet", window_sizes) == 200000

        # Test prefix match
        assert get_context_window_size("claude-3-7-opus", window_sizes) == 250000

        # Test with timestamp in name
        assert (
            get_context_window_size("claude-3-5-sonnet-20241022", window_sizes)
            == 200000
        )

        # Test fallback
        assert get_context_window_size("unknown-model", window_sizes) == 100000
