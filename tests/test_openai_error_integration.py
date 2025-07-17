"""Integration test for OpenAI tool error handling."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llmproc.common.results import ToolResult
from llmproc.program import LLMProgram
from tests.conftest import create_test_llmprocess_directly


class TestOpenAIErrorIntegration:
    """Integration tests for OpenAI tool error handling."""

    @pytest.mark.asyncio
    async def test_openai_tool_error_integration(self):
        """Test that tool errors are properly formatted for OpenAI."""
        # Create a program with a mock tool that returns an error
        program = LLMProgram(
            model_name="gpt-4o-mini",
            provider="openai",
            system_prompt="Test system prompt",
        )

        # Create process with mocked client
        process = create_test_llmprocess_directly(program=program)

        # Mock the client response to simulate a tool call
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = '{"input": "test"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_function

        mock_choice = MagicMock()
        mock_choice.message.content = "I'll use the test tool."
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        mock_response.id = "test_id"

        # Mock the client to return our tool call response
        process.client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Mock call_tool to return an error
        async def mock_call_tool(name, args):
            return ToolResult.from_error("Test tool failed")

        process.call_tool = mock_call_tool

        # Run the process
        result = await process.run("Test input", max_iterations=1)

        # Check that the tool result was formatted with ERROR prefix
        # The state should contain the formatted error message
        tool_messages = [msg for msg in process.state if msg.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["content"] == "ERROR: Test tool failed"
        assert tool_messages[0]["tool_call_id"] == "call_123"

        # Verify the tool call was tracked in the result
        assert result.tool_call_count == 1
        assert result.tool_calls[0]["tool_name"] == "test_tool"
