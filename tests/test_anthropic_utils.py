"""Tests for the Anthropic utilities module."""

import copy
from unittest.mock import MagicMock, patch

import pytest
from llmproc.common.constants import LLMPROC_MSG_ID
from llmproc.providers.anthropic_utils import (
    add_message_ids,
    add_token_efficient_header_if_needed,
    apply_cache_control,
    format_state_to_api_messages,
    format_system_prompt,
    get_context_window_size,
    is_cacheable_content,
    prepare_api_request,
)
from llmproc.providers.constants import ANTHROPIC_PROVIDERS
from llmproc.providers.utils import safe_callback
from llmproc.utils.id_utils import render_id


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
        assert is_cacheable_content({"type": "tool_result", "content": "result"}) is True
        assert is_cacheable_content({"type": "text", "text": ""}) is False
        assert is_cacheable_content({"type": "other"}) is True  # Default is True for other types


class TestMessageFormatting:
    """Tests for message formatting functions."""

    def test_add_message_ids_string_content(self):
        """Test adding message IDs to messages with string content."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = add_message_ids(messages)
        assert result[0]["content"].startswith(render_id("msg_0"))
        assert render_id("msg_1") not in result[1]["content"]

    def test_add_message_ids_list_content(self):
        """Test adding message IDs to messages with list content."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]
        result = add_message_ids(messages)
        assert result[0]["content"][0]["text"].startswith(render_id("msg_0"))
        assert render_id("msg_1") not in result[1]["content"][0]["text"]

    def test_add_message_ids_custom_msg_id(self):
        """Test that custom message IDs are used and then removed."""
        messages = [
            {"role": "user", "content": "Hello", LLMPROC_MSG_ID: "custom_id"},  # Still allow custom string IDs
            {"role": "assistant", "content": "Hi"},
        ]
        result = add_message_ids(messages)
        assert result[0]["content"].startswith(render_id("custom_id"))
        assert LLMPROC_MSG_ID not in result[0]
        assert render_id("msg_1") not in result[1]["content"]

    def test_format_and_cache_messages(self):
        """Test formatting messages and applying caching."""
        # First format the state
        state = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        # Format the messages
        formatted = format_state_to_api_messages(state)

        # Apply cache
        cached_messages, _, _ = apply_cache_control(formatted, [])

        # Verify caching
        assert len(cached_messages) == 2
        last_message = cached_messages[-1]
        assert last_message["role"] == "assistant"
        assert isinstance(last_message["content"], list)
        assert last_message["content"][0].get("cache_control") == {"type": "ephemeral"}

        # Check first message has cache too (as per our 3 message caching policy)
        assert cached_messages[0]["content"][0].get("cache_control") == {"type": "ephemeral"}

        # Verify message IDs are added
        assert isinstance(formatted[0]["content"], list)
        assert render_id("msg_0") in formatted[0]["content"][0]["text"]
        assert render_id("msg_1") not in formatted[1]["content"][0]["text"]


class TestAPIFormatting:
    """Tests for API formatting functions."""

    def test_format_system_prompt_string(self):
        """Test converting string system prompt to API format."""
        system = "Hello, I am Claude"
        result = format_system_prompt(system)

        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[0]["text"] == system

    def test_format_system_prompt_empty(self):
        """Test that empty system prompts are not modified."""
        system = ""
        result = format_system_prompt(system)
        assert result == system

    def test_format_and_cache_system_prompt(self):
        """Test formatting system prompt and applying cache."""
        system = "Hello, I am Claude"
        formatted = format_system_prompt(system)

        # Apply cache
        _, cached_system, _ = apply_cache_control([], formatted)

        assert isinstance(cached_system, list)
        assert len(cached_system) == 1
        assert cached_system[0]["type"] == "text"
        assert cached_system[0]["text"] == system
        assert cached_system[0]["cache_control"] == {"type": "ephemeral"}

    def test_prepare_api_request_with_tools(self):
        """Test prepare_api_request with tools."""
        # Create a mock process
        process = MagicMock()
        process.state = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        process.enriched_system_prompt = "You are Claude"
        process.tools = [
            {"name": "calculator", "description": "A calculator tool"},
        ]
        process.model_name = "claude-3-sonnet"
        process.api_params = {}
        process.disable_automatic_caching = False

        # Call prepare_api_request
        request = prepare_api_request(process)

        # Verify request structure
        assert "model" in request
        assert "system" in request
        assert "messages" in request
        assert "tools" in request

        # Check tools are included
        assert request["tools"] == process.tools


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
        assert "existing-feature,token-efficient-tools-2025-02-19" == result["anthropic-beta"]

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

    @patch("llmproc.providers.utils.logger")
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
        assert get_context_window_size("claude-3-5-sonnet-20241022", window_sizes) == 200000

        # Test fallback
        assert get_context_window_size("unknown-model", window_sizes) == 100000
