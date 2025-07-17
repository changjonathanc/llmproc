"""Tests for the OpenAI Response executor implementation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llmproc.common.results import ToolResult
from llmproc.program import LLMProgram
from llmproc.providers.openai_response_executor import (
    OpenAIResponseProcessExecutor,
    _normalize_responses_params,
)
from llmproc.providers.openai_utils import (
    convert_tools_to_openai_format,
    format_tool_result_for_openai,
)
from tests.conftest import create_test_llmprocess_directly


class TestOpenAIResponseExecutor:
    """Tests for the OpenAI Response executor."""

    def test_normalize_responses_params(self):
        """Test parameter normalization for Responses API."""
        # Test unsupported parameter filtering
        params = {"max_tokens": 1000, "temperature": 0.7, "top_p": 0.9}
        normalized = _normalize_responses_params(params)
        # Unsupported parameters should be removed
        assert "max_tokens" not in normalized
        assert "temperature" not in normalized
        # Supported parameters should remain
        assert "top_p" in normalized
        assert normalized["top_p"] == 0.9

    def test_normalize_responses_params_reasoning(self):
        """Test reasoning parameter handling."""
        params = {
            "reasoning_effort": "high",
            "reasoning_summary": "detailed",
            "temperature": 0.5,
        }
        normalized = _normalize_responses_params(params)
        assert "reasoning" in normalized
        assert normalized["reasoning"]["effort"] == "high"
        assert normalized["reasoning"]["summary"] == "detailed"
        assert "reasoning_effort" not in normalized
        assert "reasoning_summary" not in normalized

    def test_normalize_responses_params_default_summary(self):
        """Test default reasoning summary."""
        params = {"reasoning_effort": "medium"}
        normalized = _normalize_responses_params(params)
        assert normalized["reasoning"]["summary"] == "auto"

    def test_convert_tools(self):
        """Test tool conversion to Responses API format."""
        tools = [
            {
                "name": "calculator",
                "description": "Perform calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            }
        ]

        converted = convert_tools_to_openai_format(tools, api_type="responses")
        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["name"] == "calculator"
        assert converted[0]["description"] == "Perform calculations"
        assert converted[0]["parameters"] == tools[0]["input_schema"]

    def test_convert_tools_empty(self):
        """Test tool conversion with empty list."""
        assert convert_tools_to_openai_format(None, api_type="responses") is None
        assert convert_tools_to_openai_format([], api_type="responses") is None

    def test_format_tool_result_for_responses(self):
        """Test tool result formatting for Responses API."""
        # Test success result
        result = ToolResult.from_success("42")
        formatted = format_tool_result_for_openai(result, call_id="call_123", api_type="responses")
        assert formatted["type"] == "function_call_output"
        assert formatted["call_id"] == "call_123"
        assert formatted["output"] == "42"

    def test_format_tool_result_for_responses_error(self):
        """Test error result formatting for Responses API."""
        result = ToolResult.from_error("Division by zero")
        formatted = format_tool_result_for_openai(result, call_id="call_456", api_type="responses")
        assert formatted["type"] == "function_call_output"
        assert formatted["call_id"] == "call_456"
        assert formatted["output"] == "ERROR: Division by zero"

    def test_response_state_management(self):
        """Test response state management with complete response objects."""
        program = LLMProgram(
            model_name="o3-mini",
            provider="openai_response",
            system_prompt="Test system prompt",
        )

        process = create_test_llmprocess_directly(program=program)
        executor = OpenAIResponseProcessExecutor()

        # Mock response object
        class MockResponse:
            def __init__(self, response_id):
                self.id = response_id

        # Initially no response
        response_id, messages = executor._get_conversation_payload(process)
        assert response_id is None

        # Add response object to state
        mock_response = MockResponse("resp_123")
        executor._add_response_to_state(process, mock_response)

        assert len(process.state) == 1
        assert process.state[0]["role"] == "openai_response"
        assert process.state[0]["response_id"] == "resp_123"
        assert process.state[0]["api_type"] == "responses"
        assert process.state[0]["response"] == mock_response

        # Retrieve response ID via conversation payload
        response_id, messages = executor._get_conversation_payload(process)
        assert response_id == "resp_123"

        # Add another response
        mock_response2 = MockResponse("resp_456")
        executor._add_response_to_state(process, mock_response2)

        response_id, messages = executor._get_conversation_payload(process)
        assert response_id == "resp_456"

    @pytest.mark.asyncio
    async def test_token_counting_filters_metadata(self):
        """Test that token counting filters out response metadata."""
        program = LLMProgram(
            model_name="o3-mini",
            provider="openai_response",
            system_prompt="Test system prompt",
        )

        process = create_test_llmprocess_directly(program=program)
        executor = OpenAIResponseProcessExecutor()

        # Add some conversation state with metadata
        process.state.extend([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "response_metadata", "response_id": "resp_123"},
            {"role": "user", "content": "How are you?"},
        ])

        # Token counting should work and exclude metadata
        token_info = await executor.count_tokens(process)
        assert "input_tokens" in token_info
        assert "note" in token_info
        assert "metadata" in token_info["note"]

        # Should not include the response_metadata entry in token count
        assert "error" not in token_info

    def test_get_conversation_payload(self):
        """Test getting conversation payload for API calls."""
        program = LLMProgram(
            model_name="o3-mini",
            provider="openai_response",
            system_prompt="Test system prompt",
        )

        process = create_test_llmprocess_directly(program=program)
        executor = OpenAIResponseProcessExecutor()

        # Initially no previous response
        response_id, messages = executor._get_conversation_payload(process)
        assert response_id is None
        assert messages == []

        # Add mock response object
        class MockResponse:
            def __init__(self, response_id):
                self.id = response_id

        # Add some conversation with response and tool results
        process.state.extend([
            {"role": "user", "content": "Calculate 2+2"},
            {"role": "openai_response", "response": MockResponse("resp_123"), "response_id": "resp_123"},
            {"role": "assistant", "content": "I'll calculate that for you"},
            {"role": "tool", "content": "4", "tool_call_id": "call_456"},
            {"role": "user", "content": "What about 3+3?"},
        ])

        response_id, messages = executor._get_conversation_payload(process)
        assert response_id == "resp_123"
        assert len(messages) == 2  # Tool result + user message

        # Check tool result formatting
        assert messages[0]["type"] == "function_call_output"
        assert messages[0]["call_id"] == "call_456"
        assert messages[0]["output"] == "4"

        # Check user message (now properly formatted)
        assert messages[1] == {"type": "message", "role": "user", "content": "What about 3+3?"}
