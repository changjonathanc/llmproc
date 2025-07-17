"""Tests for shared OpenAI utility functions."""

import pytest

from llmproc.common.results import ToolResult
from llmproc.providers.openai_utils import (
    convert_tools_to_openai_format,
    format_tool_result_for_openai,
)


class TestSharedOpenAIUtils:
    """Test the shared OpenAI utility functions."""

    def test_convert_tools_chat_api(self):
        """Test tool conversion for Chat Completions API."""
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

        converted = convert_tools_to_openai_format(tools, api_type="chat")
        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert "function" in converted[0]
        assert converted[0]["function"]["name"] == "calculator"
        assert converted[0]["function"]["description"] == "Perform calculations"
        assert converted[0]["function"]["parameters"] == tools[0]["input_schema"]

    def test_convert_tools_responses_api(self):
        """Test tool conversion for Responses API."""
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
        assert "function" not in converted[0]  # Flatter structure

    def test_convert_tools_empty(self):
        """Test tool conversion with empty inputs."""
        assert convert_tools_to_openai_format(None, api_type="chat") is None
        assert convert_tools_to_openai_format([], api_type="chat") is None
        assert convert_tools_to_openai_format(None, api_type="responses") is None
        assert convert_tools_to_openai_format([], api_type="responses") is None

    def test_convert_tools_invalid_api_type(self):
        """Test tool conversion with invalid API type."""
        tools = [{"name": "test", "description": "test", "input_schema": {}}]
        with pytest.raises(ValueError, match="Unsupported api_type"):
            convert_tools_to_openai_format(tools, api_type="invalid")

    def test_format_tool_result_chat_api(self):
        """Test tool result formatting for Chat Completions API."""
        result = ToolResult.from_success("42")
        formatted = format_tool_result_for_openai(result, api_type="chat")
        assert formatted == "42"

        # Test error formatting
        error_result = ToolResult.from_error("Division by zero")
        formatted_error = format_tool_result_for_openai(error_result, api_type="chat")
        assert formatted_error == "ERROR: Division by zero"

    def test_format_tool_result_responses_api(self):
        """Test tool result formatting for Responses API."""
        result = ToolResult.from_success("42")
        formatted = format_tool_result_for_openai(result, call_id="call_123", api_type="responses")
        assert formatted["type"] == "function_call_output"
        assert formatted["call_id"] == "call_123"
        assert formatted["output"] == "42"

        # Test error formatting
        error_result = ToolResult.from_error("Division by zero")
        formatted_error = format_tool_result_for_openai(error_result, call_id="call_456", api_type="responses")
        assert formatted_error["type"] == "function_call_output"
        assert formatted_error["call_id"] == "call_456"
        assert formatted_error["output"] == "ERROR: Division by zero"

    def test_format_tool_result_responses_api_missing_call_id(self):
        """Test that Responses API requires call_id."""
        result = ToolResult.from_success("42")
        with pytest.raises(ValueError, match="call_id is required for Responses API"):
            format_tool_result_for_openai(result, api_type="responses")

    def test_format_tool_result_invalid_api_type(self):
        """Test tool result formatting with invalid API type."""
        result = ToolResult.from_success("42")
        with pytest.raises(ValueError, match="Unsupported api_type"):
            format_tool_result_for_openai(result, api_type="invalid")
