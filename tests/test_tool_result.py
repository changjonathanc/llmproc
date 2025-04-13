"""Tests for the ToolResult class."""

import json
from typing import Any

import pytest

from llmproc.common.results import ToolResult


def test_tool_result_init():
    """Test basic initialization of ToolResult."""
    # Test with string content
    result = ToolResult("Test content")
    assert result.content == "Test content"
    assert result.is_error is False

    # Test with dictionary content
    dict_content = {"key": "value"}
    result = ToolResult(dict_content)
    assert result.content == dict_content
    assert result.is_error is False

    # Test with error flag
    error_result = ToolResult("Error message", is_error=True)
    assert error_result.content == "Error message"
    assert error_result.is_error is True


def test_tool_result_from_success():
    """Test the from_success factory method."""
    result = ToolResult.from_success("Success content")
    assert result.content == "Success content"
    assert result.is_error is False


def test_tool_result_from_error():
    """Test the from_error factory method."""
    result = ToolResult.from_error("Error message")
    assert result.content == "Error message"
    assert result.is_error is True


def test_tool_result_to_dict():
    """Test conversion to dictionary."""
    # String content
    result = ToolResult("Test content")
    assert result.to_dict() == {"content": "Test content", "is_error": False}

    # Dictionary content - should be JSON serialized
    dict_content = {"key": "value"}
    result = ToolResult(dict_content)
    assert result.to_dict() == {"content": json.dumps(dict_content), "is_error": False}

    # Error result
    error_result = ToolResult.from_error("Error message")
    assert error_result.to_dict() == {"content": "Error message", "is_error": True}


def test_tool_result_with_non_serializable_content():
    """Test handling of non-JSON-serializable content."""

    # Create a non-serializable object
    class NonSerializable:
        def __str__(self):
            return "NonSerializable object"

    non_serializable = NonSerializable()

    # Test with non-serializable as content
    result = ToolResult(non_serializable)
    dict_result = result.to_dict()
    assert dict_result["content"] == str(non_serializable)
    assert dict_result["is_error"] is False

    # Test with non-serializable in dictionary
    result = ToolResult({"obj": non_serializable})
    dict_result = result.to_dict()
    assert (
        "NonSerializable" in dict_result["content"]
        or "object" in dict_result["content"]
    )  # Should contain some reference to our object
    assert dict_result["is_error"] is False


def test_tool_result_str():
    """Test string representation."""
    result = ToolResult("Test content")
    assert str(result) == "ToolResult(content=Test content, is_error=False)"

    error_result = ToolResult.from_error("Error message")
    assert str(error_result) == "ToolResult(content=Error message, is_error=True)"
