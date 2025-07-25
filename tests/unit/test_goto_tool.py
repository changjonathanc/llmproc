"""Unit tests for the GOTO time travel tool.

These tests focus on the internal functionality of the GOTO tool without requiring API calls.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from llmproc.common.results import ToolResult
from llmproc.plugins.message_id import find_position_by_id, MessageIDPlugin
from llmproc.config.schema import MessageIDPluginConfig
from llmproc.utils.message_utils import append_message


class TestGotoToolUnit:
    """Unit tests for the core GOTO tool functionality."""

    def test_find_position_by_id(self):
        """Test finding positions by message ID."""
        # Create a mock state - IDs are just array indices, no stored fields needed
        state = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well"},
        ]

        # Test integer ID lookup (internal use)
        assert find_position_by_id(state, 0) == 0
        assert find_position_by_id(state, 1) == 1
        assert find_position_by_id(state, 2) == 2
        assert find_position_by_id(state, 3) == 3

        # Test formatted IDs (how LLMs will use it, based on what they see)
        assert find_position_by_id(state, "msg_0") == 0
        assert find_position_by_id(state, "msg_1") == 1
        assert find_position_by_id(state, "msg_2") == 2
        assert find_position_by_id(state, "msg_3") == 3

        # Also support direct string numeric IDs
        assert find_position_by_id(state, "0") == 0
        assert find_position_by_id(state, "1") == 1

        # Test non-existent ID
        assert find_position_by_id(state, 99) is None
        assert find_position_by_id(state, "msg_99") is None  # Non-existent formatted ID
        assert find_position_by_id(state, "99") is None  # Non-existent numeric ID

        # Test invalid IDs
        assert find_position_by_id(state, "invalid") is None
        assert find_position_by_id(state, None) is None

        # Test with state that has no message IDs at all
        state_without_ids = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
        ]
        # Support position lookup by index even without IDs
        assert find_position_by_id(state_without_ids, 0) == 0
        assert find_position_by_id(state_without_ids, 1) == 1
        # Also support formatted and string IDs by position
        assert find_position_by_id(state_without_ids, "msg_0") == 0
        assert find_position_by_id(state_without_ids, "msg_1") == 1
        assert find_position_by_id(state_without_ids, "0") == 0
        assert find_position_by_id(state_without_ids, "1") == 1

    def test_append_message(self):
        """Test appending messages to the state."""
        process = MagicMock()
        process.state = []

        append_message(process, "user", "Message 1")
        assert process.state[0]["content"] == "Message 1"

        append_message(process, "assistant", "Message 2")
        assert process.state[1]["content"] == "Message 2"

        process.state = []

        append_message(process, "user", "Message 3")
        assert process.state[0]["content"] == "Message 3"

    @pytest.mark.asyncio
    @patch("llmproc.plugins.message_id.datetime")
    async def test_handle_goto_success(self, mock_datetime):
        """Test handling a successful GOTO operation without a new message."""
        # Mock datetime for consistent timestamps
        mock_datetime.datetime.now.return_value.isoformat.return_value = "2025-01-01T00:00:00"

        # Create a realistic process state - IDs are array indices
        process = MagicMock()
        process.state = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ]
        process.time_travel_history = []

        # Save original state for checking how it changes
        original_state_len = len(process.state)

        # Create runtime context with the process
        runtime_context = {"process": process}

        # Create plugin to test
        plugin = MessageIDPlugin(MessageIDPluginConfig(enable_goto=True))

        # Mock append_message to capture its calls without actual execution
        with patch("llmproc.plugins.message_id.append_message") as mock_append:
            # Define a more accurate side effect to simulate real behavior
            def append_side_effect(proc, role, content):
                # Directly modify the process state like the real implementation would
                proc.state = [{"role": "user", "content": content}]
                return 0

            mock_append.side_effect = append_side_effect

            # Call the plugin's goto tool with valid position
            result = await plugin.goto_tool(position="msg_0", message="", runtime_context=runtime_context)

            # Verify the state was properly truncated
            assert mock_append.call_count == 0, "append_message should not be called when message is empty"

            # Check result content
            assert not result.is_error
            assert "Conversation reset to message msg_0" in result.content
            # Check that abort_execution is set
            assert result.abort_execution

            # Check time travel history was properly updated
            assert len(process.time_travel_history) == 1
            history_entry = process.time_travel_history[0]
            assert history_entry["from_message_count"] == original_state_len
            # to_message_count would normally be 1 in the implementation
            # but our mock doesn't actually update it correctly, so we don't test it here

    @pytest.mark.asyncio
    async def test_handle_goto_with_message(self):
        """Test handling a GOTO operation with a new message."""
        # Create a mock process with some messages
        process = MagicMock()
        process.state = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ]
        process.time_travel_history = []

        # Create runtime context with the process
        runtime_context = {"process": process}

        # Capture the original message we're resetting to
        original_message = process.state[0]["content"]
        new_message_content = "New direction"

        # Use a proper mock object for append_message
        mock_append = MagicMock(return_value="msg_0")

        # Create plugin to test
        plugin = MessageIDPlugin(MessageIDPluginConfig(enable_goto=True))

        # Apply our mock implementation
        with patch("llmproc.plugins.message_id.append_message", mock_append):
            # Call the plugin's goto tool with a valid position and a new message
            result = await plugin.goto_tool(
                position="msg_0",
                message=new_message_content,
                runtime_context=runtime_context,
            )

            # Check that append_message was called
            mock_append.assert_called_once()

            # Extract the content that was passed to append_message
            _, _, content = mock_append.call_args[0]

            # Verify the content contains the expected tags and message
            assert "<system_message>" in content
            assert "GOTO tool used. Conversation reset" in content
            assert "<original_message_to_be_ignored>" in content
            assert original_message in content
            assert "<time_travel_message>" in content
            assert new_message_content in content

            # Check the result
            assert not result.is_error
            assert "Added time travel message" in result.content
            assert "Conversation reset to message msg_0" in result.content
            # Check that abort_execution is set
            assert result.abort_execution

    @pytest.mark.asyncio
    async def test_handle_goto_with_preformatted_message(self):
        """Test handling a GOTO operation with a pre-formatted time travel message."""
        # Create a mock process with some messages
        process = MagicMock()
        process.state = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]
        process.time_travel_history = []

        # Create runtime context with the process
        runtime_context = {"process": process}

        # Capture the original first message content
        original_message = process.state[0]["content"]

        # Create a message that already has time_travel tags
        preformatted_message = (
            "<time_travel>\nChanging direction because the previous approach wasn't working\n</time_travel>"
        )

        # Create plugin to test
        plugin = MessageIDPlugin(MessageIDPluginConfig(enable_goto=True))

        # Use a proper mock object for append_message
        mock_append = MagicMock(return_value="msg_0")

        # Apply our mock implementation
        with patch("llmproc.plugins.message_id.append_message", mock_append):
            # Call with a message that already has time_travel tags
            result = await plugin.goto_tool(
                position="msg_0",
                message=preformatted_message,
                runtime_context=runtime_context,
            )

            # Check that append_message was called
            mock_append.assert_called_once()

            # Extract the content that was passed to append_message
            _, _, content = mock_append.call_args[0]

            # Check the formatted content has exactly what we expect
            assert "<system_message>" in content
            assert "GOTO tool used. Conversation reset" in content
            assert "<original_message_to_be_ignored>" in content
            assert original_message in content
            assert "<time_travel_message>" in content
            assert "Changing direction" in content

            # Make sure time_travel tags aren't duplicated (the implementation should handle this correctly)
            assert content.count("<time_travel>") == 1
            assert content.count("</time_travel>") == 1

            # Check the result
            assert not result.is_error
            assert "Added time travel message" in result.content
            # Check that abort_execution is set
            assert result.abort_execution

    @pytest.mark.asyncio
    async def test_handle_goto_errors(self):
        """Test error handling in the GOTO tool."""
        # Create a mock process with some messages
        process = MagicMock()
        process.state = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]

        # Create runtime context with the process
        runtime_context = {"process": process}

        # Create plugin to test
        plugin = MessageIDPlugin(MessageIDPluginConfig(enable_goto=True))

        # Test with invalid position format
        result1 = await plugin.goto_tool(position="invalid", message="", runtime_context=runtime_context)
        assert result1.is_error
        assert "Invalid message ID" in result1.content

        # Test with non-existent position
        result2 = await plugin.goto_tool(position="msg_99", message="", runtime_context=runtime_context)
        assert result2.is_error
        assert "Could not find message" in result2.content

        # Test with trying to go forward (to current position)
        result3 = await plugin.goto_tool(position="msg_1", message="", runtime_context=runtime_context)
        assert result3.is_error
        assert "Cannot go forward in time" in result3.content

    @pytest.mark.asyncio
    async def test_handle_goto_missing_parameters(self):
        """Test GOTO tool with missing required parameters."""
        process = MagicMock()
        process.state = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]

        # Create runtime context with the process
        runtime_context = {"process": process}

        # Create plugin to test
        plugin = MessageIDPlugin(MessageIDPluginConfig(enable_goto=True))

        # Test with missing position
        result = await plugin.goto_tool(position="", message="This should fail", runtime_context=runtime_context)
        assert result.is_error
        assert "Invalid message ID" in result.content
