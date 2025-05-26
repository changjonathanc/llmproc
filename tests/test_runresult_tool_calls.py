"""Test suite for RunResult tool_calls tracking.

This tests the functionality of the RunResult.add_tool_call() method.
"""

import pytest
from llmproc.common.results import RunResult


def test_runresult_records_tool_call():
    """Test that RunResult.add_tool_call() works correctly."""
    # Create a RunResult
    result = RunResult()

    # Initial state should be empty
    assert len(result.tool_calls) == 0

    # Add a tool call
    result.add_tool_call("test_tool", {"param1": "value1"})

    # Check the tool_calls collection was updated
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool_name"] == "test_tool"
    assert isinstance(result.tool_calls[0]["args"], dict)
    assert result.tool_calls[0]["args"]["param1"] == "value1"

    # Test with multiple tool calls
    result.add_tool_call("another_tool")
    assert len(result.tool_calls) == 2
    assert result.tool_calls[1]["tool_name"] == "another_tool"


def test_total_interactions_property():
    """Test the total_interactions property which relies on tool_calls."""
    result = RunResult()

    # Add a tool call
    result.add_tool_call("test_tool")

    # Add an API call
    result.add_api_call({"model": "test", "usage": {"input_tokens": 10}})

    # total_interactions should be the sum of API calls and tool calls
    assert result.total_interactions == 2

    # Add another tool call
    result.add_tool_call("another_tool")

    # total_interactions should update
    assert result.total_interactions == 3


def test_total_interactions_counts_calls():
    """Test that total_interactions properly counts both API calls and tool calls."""
    # Create a RunResult
    result = RunResult()

    # Add API calls
    result.add_api_call({"model": "test1"})
    result.add_api_call({"model": "test2"})

    # Verify API calls count
    assert result.api_calls == 2

    # Add tool calls
    result.add_tool_call("tool1")
    result.add_tool_call("tool2")
    result.add_tool_call("tool3")

    # Verify tool calls counts
    assert len(result.tool_calls) == 3

    # Verify total interactions counts both API calls and tool calls
    assert result.total_interactions == 5  # 2 API calls + 3 tool calls
