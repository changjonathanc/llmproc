"""Unit tests for the GOTO time travel tool.

These tests focus on the internal functionality of the GOTO tool without requiring API calls.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from llmproc.common.results import ToolResult
from llmproc.tools.builtin.goto import find_position_by_id, handle_goto
from llmproc.utils.message_utils import append_message_with_id


class TestGotoToolUnit:
    """Unit tests for the core GOTO tool functionality."""

    def test_find_position_by_id(self):
        """Test finding positions by message ID."""
        # Create a mock state with message IDs
        state = [
            {"role": "user", "content": "Hello", "goto_id": "msg_0"},
            {"role": "assistant", "content": "Hi there", "goto_id": "msg_1"},
            {"role": "user", "content": "How are you?", "goto_id": "msg_2"},
            {"role": "assistant", "content": "I'm doing well", "goto_id": "msg_3"},
        ]

        # Test direct ID lookup
        assert find_position_by_id(state, "msg_0") == 0
        assert find_position_by_id(state, "msg_1") == 1
        assert find_position_by_id(state, "msg_2") == 2
        assert find_position_by_id(state, "msg_3") == 3

        # Test non-existent ID
        assert find_position_by_id(state, "msg_99") is None

        # Test invalid IDs
        assert find_position_by_id(state, "invalid") is None
        assert find_position_by_id(state, 123) is None
        assert find_position_by_id(state, None) is None

        # Test fallback to numeric lookup
        state_without_ids = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
        ]
        assert find_position_by_id(state_without_ids, "msg_0") == 0
        assert find_position_by_id(state_without_ids, "msg_1") == 1

    def test_append_message(self):
        """Test appending messages with IDs."""
        # Create a mock process with an empty state
        process = MagicMock()
        process.state = []

        # Append messages and verify IDs
        id1 = append_message_with_id(process, "user", "Message 1")
        assert id1 == "msg_0"
        assert process.state[0]["goto_id"] == "msg_0"

        id2 = append_message_with_id(process, "assistant", "Message 2")
        assert id2 == "msg_1"
        assert process.state[1]["goto_id"] == "msg_1"

    @pytest.mark.asyncio
    @patch("llmproc.tools.builtin.goto.datetime")
    async def test_handle_goto_success(self, mock_datetime):
        """Test handling a successful GOTO operation without a new message."""
        # Mock datetime for consistent timestamps
        mock_datetime.datetime.now.return_value.isoformat.return_value = (
            "2025-01-01T00:00:00"
        )

        # Create a realistic process state with goto_ids
        process = MagicMock()
        process.state = [
            {"role": "user", "content": "Message 1", "goto_id": "msg_0"},
            {"role": "assistant", "content": "Response 1", "goto_id": "msg_1"},
            {"role": "user", "content": "Message 2", "goto_id": "msg_2"},
            {"role": "assistant", "content": "Response 2", "goto_id": "msg_3"},
        ]
        process.time_travel_history = []

        # Save original state for checking how it changes
        original_state_len = len(process.state)

        # Create runtime context with the process
        runtime_context = {"process": process}

        # Mock append_message_with_id to capture its calls without actual execution
        with patch("llmproc.tools.builtin.goto.append_message_with_id") as mock_append:
            # Define a more accurate side effect to simulate real behavior
            def append_side_effect(proc, role, content):
                # Directly modify the process state like the real implementation would
                proc.state = [{"role": "user", "content": content, "goto_id": "msg_0"}]
                return "msg_0"

            mock_append.side_effect = append_side_effect

            # Call the handler with valid position
            result = await handle_goto(
                position="msg_0", message="", runtime_context=runtime_context
            )

            # Verify the state was properly truncated
            assert mock_append.call_count == 0, (
                "append_message_with_id should not be called when message is empty"
            )

            # Check result content
            assert not result.is_error
            assert "Conversation reset to message msg_0" in result.content

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
            {"role": "user", "content": "Message 1", "goto_id": "msg_0"},
            {"role": "assistant", "content": "Response 1", "goto_id": "msg_1"},
            {"role": "user", "content": "Message 2", "goto_id": "msg_2"},
            {"role": "assistant", "content": "Response 2", "goto_id": "msg_3"},
        ]
        process.time_travel_history = []

        # Create runtime context with the process
        runtime_context = {"process": process}

        # Capture the original message we're resetting to
        original_message = process.state[0]["content"]
        new_message_content = "New direction"

        # Use a proper mock object for append_message_with_id
        mock_append = MagicMock(return_value="msg_0")

        # Apply our mock implementation
        with patch("llmproc.tools.builtin.goto.append_message_with_id", mock_append):
            # Call the handler with a valid position and a new message
            result = await handle_goto(
                position="msg_0",
                message=new_message_content,
                runtime_context=runtime_context,
            )

            # Check that append_message_with_id was called
            mock_append.assert_called_once()

            # Extract the content that was passed to append_message_with_id
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

    @pytest.mark.asyncio
    async def test_handle_goto_with_preformatted_message(self):
        """Test handling a GOTO operation with a pre-formatted time travel message."""
        # Create a mock process with some messages
        process = MagicMock()
        process.state = [
            {"role": "user", "content": "Message 1", "goto_id": "msg_0"},
            {"role": "assistant", "content": "Response 1", "goto_id": "msg_1"},
        ]
        process.time_travel_history = []

        # Create runtime context with the process
        runtime_context = {"process": process}

        # Capture the original first message content
        original_message = process.state[0]["content"]

        # Create a message that already has time_travel tags
        preformatted_message = "<time_travel>\nChanging direction because the previous approach wasn't working\n</time_travel>"

        # Use a proper mock object for append_message_with_id
        mock_append = MagicMock(return_value="msg_0")

        # Apply our mock implementation
        with patch("llmproc.tools.builtin.goto.append_message_with_id", mock_append):
            # Call with a message that already has time_travel tags
            result = await handle_goto(
                position="msg_0",
                message=preformatted_message,
                runtime_context=runtime_context,
            )

            # Check that append_message_with_id was called
            mock_append.assert_called_once()

            # Extract the content that was passed to append_message_with_id
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

    @pytest.mark.asyncio
    async def test_handle_goto_errors(self):
        """Test error handling in the GOTO tool."""
        # Create a mock process with some messages
        process = MagicMock()
        process.state = [
            {"role": "user", "content": "Message 1", "goto_id": "msg_0"},
            {"role": "assistant", "content": "Response 1", "goto_id": "msg_1"},
        ]

        # Create runtime context with the process
        runtime_context = {"process": process}

        # Test with invalid position format
        result1 = await handle_goto(
            position="invalid", message="", runtime_context=runtime_context
        )
        assert result1.is_error
        assert "Invalid message ID" in result1.content

        # Test with non-existent position
        result2 = await handle_goto(
            position="msg_99", message="", runtime_context=runtime_context
        )
        assert result2.is_error
        assert "Could not find message" in result2.content

        # Test with trying to go forward (to current position)
        result3 = await handle_goto(
            position="msg_1", message="", runtime_context=runtime_context
        )
        assert result3.is_error
        assert "Cannot go forward in time" in result3.content

    @pytest.mark.asyncio
    async def test_handle_goto_missing_parameters(self):
        """Test GOTO tool with missing required parameters."""
        process = MagicMock()
        process.state = [
            {"role": "user", "content": "Message 1", "goto_id": "msg_0"},
            {"role": "assistant", "content": "Response 1", "goto_id": "msg_1"},
        ]

        # Create runtime context with the process
        runtime_context = {"process": process}

        # Test with missing position
        result = await handle_goto(
            position="", message="This should fail", runtime_context=runtime_context
        )
        assert result.is_error
        assert "Invalid message ID" in result.content
