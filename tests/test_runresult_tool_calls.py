"""Test suite for RunResult tool_calls tracking.

This tests the fix for the issue where RunResult.add_tool_call() needed
to update both tool_call_infos and tool_calls for backward compatibility.
"""

import pytest

from llmproc.common.results import RunResult


def test_runresult_add_tool_call():
    """Test that RunResult.add_tool_call() updates both tool_call_infos and tool_calls."""
    # Create a RunResult
    result = RunResult()

    # Initial state should be empty
    assert len(result.tool_calls) == 0
    assert len(result.tool_call_infos) == 0

    # Add a tool call
    tool_info = {
        "type": "tool_call",
        "tool_name": "test_tool",
        "args": {"param1": "value1"},
    }
    result.add_tool_call(tool_info)

    # Both tool_calls and tool_call_infos should be updated
    assert len(result.tool_calls) == 1
    assert len(result.tool_call_infos) == 1

    # Check the tool_calls entry format (name, args, result)
    assert result.tool_calls[0][0] == "test_tool"  # Name should match
    assert isinstance(result.tool_calls[0][1], dict)  # Args should be a dict
    assert result.tool_calls[0][2] is None  # Result is not captured in add_tool_call

    # Check the tool_call_infos entry
    assert result.tool_call_infos[0] == tool_info

    # Test with multiple tool calls
    result.add_tool_call({"type": "tool_call", "tool_name": "another_tool"})
    assert len(result.tool_calls) == 2
    assert len(result.tool_call_infos) == 2
    assert result.tool_calls[1][0] == "another_tool"


def test_runresult_tool_calls_property():
    """Test the total_interactions property which relies on tool_calls."""
    result = RunResult()

    # Add a tool call
    result.add_tool_call({"type": "tool_call", "tool_name": "test_tool"})

    # Add an API call
    result.add_api_call({"model": "test", "usage": {"input_tokens": 10}})

    # total_interactions should be the sum of API calls and tool calls
    assert result.total_interactions == 2

    # Add another tool call
    result.add_tool_call({"type": "tool_call", "tool_name": "another_tool"})

    # total_interactions should update
    assert result.total_interactions == 3


def test_runresult_missing_tool_name():
    """Test adding a tool call without a tool_name."""
    result = RunResult()

    # Add a tool call without a tool_name
    result.add_tool_call({"type": "tool_call", "args": {"param1": "value1"}})

    # tool_call_infos should be updated but not tool_calls
    assert len(result.tool_call_infos) == 1
    assert len(result.tool_calls) == 0  # tool_calls requires tool_name


def test_total_interactions_counts_both_collections():
    """Test that total_interactions properly counts both API calls and tool calls."""
    # Create a RunResult
    result = RunResult()

    # Add API calls
    result.add_api_call({"model": "test1"})
    result.add_api_call({"model": "test2"})

    # Verify API calls count
    assert result.api_calls == 2

    # Add tool calls using both methods
    # 1. Direct addition to tool_calls (old style)
    result.tool_calls.append(("old_tool", {}, None))

    # 2. Using add_tool_call which populates both collections (new style)
    result.add_tool_call({"type": "tool_call", "tool_name": "new_tool"})

    # Verify tool calls counts
    assert len(result.tool_calls) == 2  # Both old and new style
    assert len(result.tool_call_infos) == 1  # Only new style

    # Verify total interactions counts both API calls and tool calls
    assert result.total_interactions == 4  # 2 API calls + 2 tool calls
