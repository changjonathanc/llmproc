"""Tests for graceful tool error handling."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from llmproc.common.results import ToolResult
from llmproc.tools import ToolManager
from llmproc.tools.core import Tool
from llmproc.common.metadata import get_tool_meta


@pytest.fixture
async def tool_manager():
    """Create a tool manager with a test tool."""
    manager = ToolManager()

    # Define a simple test tool handler
    async def test_tool_handler(**kwargs):
        return ToolResult.from_success("Test tool success")

    # Register the tool
    manager.runtime_registry.register_tool_obj(
        Tool(
            handler=test_tool_handler,
            schema={"name": "test_tool", "description": "A test tool"},
            meta=get_tool_meta(test_tool_handler),
        )
    )

    # Register the tool by handler callable (not by name)
    await manager.register_tools([test_tool_handler])

    return manager


@pytest.mark.asyncio
async def test_call_valid_tool(tool_manager):
    """Test calling a valid tool."""
    result = await tool_manager.call_tool("test_tool", {})
    assert isinstance(result, ToolResult)
    assert result.content == "Test tool success"
    assert not result.is_error


@pytest.mark.asyncio
async def test_call_nonexistent_tool(tool_manager):
    """Test calling a nonexistent tool returns an error ToolResult."""
    result = await tool_manager.call_tool("nonexistent_tool", {})
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "nonexistent_tool" in result.content
    assert "list_tools" in result.content


@pytest.mark.asyncio
async def test_tool_execution_error(tool_manager):
    """Test error during tool execution returns an error ToolResult."""

    # Register a tool that raises an exception
    async def error_tool_handler(**kwargs):
        raise ValueError("Test error")

    tool_manager.runtime_registry.register_tool_obj(
        Tool(
            handler=error_tool_handler,
            schema={"name": "error_tool", "description": "A tool that errors"},
            meta=get_tool_meta(error_tool_handler),
        )
    )

    # Register the error tool by handler callable
    await tool_manager.register_tools([error_tool_handler])

    result = await tool_manager.call_tool("error_tool", {})
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert result.content.startswith("Error:")
    assert "Test error" in result.content
