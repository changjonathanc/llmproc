"""Unit tests for the GOTO time travel tool.

These tests focus on the internal functionality of the GOTO tool without requiring API calls.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from llmproc.tools.builtin.goto import find_position_by_id, handle_goto
from llmproc.utils.message_utils import append_message_with_id
from llmproc.common.results import ToolResult


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
        """Test handling a successful GOTO operation."""
        # Mock datetime to have a predictable timestamp
        mock_datetime.datetime.now.return_value.isoformat.return_value = "2025-01-01T00:00:00"
        
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
        
        # Call the handler with a valid position and runtime context
        result = await handle_goto(position="msg_1", message="", runtime_context=runtime_context)
        
        # Check that the state was truncated
        assert len(process.state) == 2
        assert process.state[-1]["goto_id"] == "msg_1"
        
        # Check that the time travel history was updated
        assert len(process.time_travel_history) == 1
        assert process.time_travel_history[0]["from_message_count"] == 4
        assert process.time_travel_history[0]["to_message_count"] == 2
        
        # Check the result
        assert not result.is_error
        assert "Conversation reset to message msg_1" in result.content
    
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
        
        # Call the handler with a valid position and a new message
        result = await handle_goto(position="msg_0", message="New direction", runtime_context=runtime_context)
        
        # Check that the state was truncated and new message added
        assert len(process.state) == 2
        assert process.state[0]["goto_id"] == "msg_0"
        assert process.state[1]["role"] == "user"
        
        # Verify the content structure
        assert "[SYSTEM NOTE: Conversation reset" in process.state[1]["content"]
        assert "<time_travel>" in process.state[1]["content"]
        assert "New direction" in process.state[1]["content"]
        assert "</time_travel>" in process.state[1]["content"]
        
        # Check the result
        assert not result.is_error
        assert "Added time travel message" in result.content
        
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
        
        # Call with a message that already has time_travel tags
        preformatted_message = "<time_travel>\nChanging direction because the previous approach wasn't working\n</time_travel>"
        result = await handle_goto(position="msg_0", message=preformatted_message, runtime_context=runtime_context)
        
        # Check that the message is properly formatted with only one set of tags
        assert len(process.state) == 2
        assert "[SYSTEM NOTE:" in process.state[1]["content"]
        assert "<time_travel>" in process.state[1]["content"]
        assert "Changing direction" in process.state[1]["content"]
        
        # Make sure tags weren't duplicated
        assert process.state[1]["content"].count("<time_travel>") == 1
        assert process.state[1]["content"].count("</time_travel>") == 1
    
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
        result1 = await handle_goto(position="invalid", message="", runtime_context=runtime_context)
        assert result1.is_error
        assert "Invalid message ID" in result1.content
        
        # Test with non-existent position
        result2 = await handle_goto(position="msg_99", message="", runtime_context=runtime_context)
        assert result2.is_error
        assert "Could not find message" in result2.content
        
        # Test with trying to go forward (to current position)
        result3 = await handle_goto(position="msg_1", message="", runtime_context=runtime_context)
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
        result = await handle_goto(position="", message="This should fail", runtime_context=runtime_context)
        assert result.is_error
        assert "Invalid message ID" in result.content