"""Tests for graceful tool error handling."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from llmproc.tools import ToolRegistry
from llmproc.tools.tool_result import ToolResult


@pytest.fixture
def tool_registry():
    """Create a tool registry with a test tool."""
    registry = ToolRegistry()
    
    # Define a simple test tool handler
    async def test_tool_handler(args):
        return ToolResult.from_success("Test tool success")
    
    # Register the tool
    registry.register_tool(
        "test_tool",
        test_tool_handler,
        {"name": "test_tool", "description": "A test tool"}
    )
    
    return registry


@pytest.mark.asyncio
async def test_call_valid_tool(tool_registry):
    """Test calling a valid tool."""
    result = await tool_registry.call_tool("test_tool", {})
    assert isinstance(result, ToolResult)
    assert result.content == "Test tool success"
    assert not result.is_error


@pytest.mark.asyncio
async def test_call_nonexistent_tool(tool_registry):
    """Test calling a nonexistent tool returns an error ToolResult."""
    result = await tool_registry.call_tool("nonexistent_tool", {})
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "not found" in result.content
    assert "Available tools: test_tool" in result.content


@pytest.mark.asyncio
async def test_tool_execution_error(tool_registry):
    """Test error during tool execution returns an error ToolResult."""
    # Register a tool that raises an exception
    async def error_tool_handler(args):
        raise ValueError("Test error")
    
    tool_registry.register_tool(
        "error_tool",
        error_tool_handler,
        {"name": "error_tool", "description": "A tool that errors"}
    )
    
    result = await tool_registry.call_tool("error_tool", {})
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "Error executing tool" in result.content
    assert "Test error" in result.content