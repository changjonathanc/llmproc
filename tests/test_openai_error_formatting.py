"""Tests for OpenAI tool error formatting."""

import pytest

from llmproc.common.results import ToolResult
from llmproc.providers.openai_utils import format_tool_result_for_openai


class TestOpenAIErrorFormatting:
    """Tests for the OpenAI tool error formatting function."""

    def test_format_successful_tool_result(self):
        """Test formatting a successful tool result."""
        result = ToolResult.from_success("Operation completed successfully")
        formatted = format_tool_result_for_openai(result)
        assert formatted == "Operation completed successfully"

    def test_format_error_tool_result(self):
        """Test formatting an error tool result."""
        result = ToolResult.from_error("File not found")
        formatted = format_tool_result_for_openai(result)
        assert formatted == "ERROR: File not found"

    def test_format_empty_content(self):
        """Test formatting with empty content."""
        result = ToolResult(content="", is_error=False)
        formatted = format_tool_result_for_openai(result)
        assert formatted == ""

    def test_format_empty_error_content(self):
        """Test formatting an error with empty content."""
        result = ToolResult(content="", is_error=True)
        formatted = format_tool_result_for_openai(result)
        assert formatted == "ERROR: "

    def test_format_none_content(self):
        """Test formatting with None content."""
        result = ToolResult(content=None, is_error=False)
        formatted = format_tool_result_for_openai(result)
        assert formatted == ""

    def test_format_none_error_content(self):
        """Test formatting an error with None content."""
        result = ToolResult(content=None, is_error=True)
        formatted = format_tool_result_for_openai(result)
        assert formatted == "ERROR: "

    def test_format_dict_content(self):
        """Test formatting with dictionary content."""
        result = ToolResult.from_success({"status": "ok", "data": [1, 2, 3]})
        formatted = format_tool_result_for_openai(result)
        # Should convert dict to JSON string
        assert '"status": "ok"' in formatted
        assert '"data": [1, 2, 3]' in formatted

    def test_format_dict_error_content(self):
        """Test formatting an error with dictionary content."""
        result = ToolResult(content={"error": "Invalid input", "code": 400}, is_error=True)
        formatted = format_tool_result_for_openai(result)
        assert formatted.startswith("ERROR: ")
        assert '"error": "Invalid input"' in formatted
        assert '"code": 400' in formatted
