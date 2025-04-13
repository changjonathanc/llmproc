"""Tests for MCP integration functions."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from llmproc.common.results import ToolResult
from llmproc.tools.mcp.integration import (
    initialize_mcp_tools,
    register_mcp_tool,
    register_runtime_mcp_tools,
)
from llmproc.tools.tool_registry import ToolRegistry


async def dummy_handler(args):
    """Simple dummy handler for testing."""
    return ToolResult.from_success("Test result")


def test_register_mcp_tool():
    """Test registering an MCP tool to the runtime registry."""
    mcp_registry = ToolRegistry()
    runtime_registry = ToolRegistry()

    # Register a test tool in the MCP registry
    test_schema = {
        "name": "mcp_test_tool",
        "description": "MCP test tool",
        "input_schema": {"type": "object", "properties": {}},
    }
    mcp_registry.register_tool("mcp_test_tool", dummy_handler, test_schema)

    # Register the MCP tool to the runtime registry
    result = register_mcp_tool(mcp_registry, runtime_registry, "mcp_test_tool")

    # Verify registration succeeded
    assert result is True
    assert "mcp_test_tool" in runtime_registry.tool_handlers
    assert runtime_registry.tool_handlers["mcp_test_tool"] is dummy_handler

    # Verify schema was copied
    runtime_schemas = runtime_registry.get_definitions()
    assert len(runtime_schemas) == 1
    assert runtime_schemas[0]["name"] == "mcp_test_tool"
    assert runtime_schemas[0]["description"] == "MCP test tool"

    # Test with non-existent tool
    result = register_mcp_tool(mcp_registry, runtime_registry, "nonexistent")
    assert result is False


def test_register_runtime_mcp_tools():
    """Test registering all MCP tools to runtime registry."""
    mcp_registry = ToolRegistry()
    runtime_registry = ToolRegistry()
    enabled_tools = ["existing_tool"]

    # Register multiple test tools in MCP registry
    for i in range(3):
        test_schema = {
            "name": f"mcp_tool_{i}",
            "description": f"MCP tool {i}",
            "input_schema": {"type": "object", "properties": {}},
        }
        mcp_registry.register_tool(f"mcp_tool_{i}", dummy_handler, test_schema)

    # Register MCP tools to runtime registry
    count = register_runtime_mcp_tools(mcp_registry, runtime_registry, enabled_tools)

    # Verify registration succeeded
    assert count == 3

    # Verify tools are in runtime registry
    for i in range(3):
        assert f"mcp_tool_{i}" in runtime_registry.tool_handlers

    # Verify tools were added to enabled_tools
    assert len(enabled_tools) == 4  # existing_tool + 3 new MCP tools
    for i in range(3):
        assert f"mcp_tool_{i}" in enabled_tools

    # Test with empty MCP registry
    empty_mcp_registry = ToolRegistry()
    count = register_runtime_mcp_tools(
        empty_mcp_registry, runtime_registry, enabled_tools
    )
    assert count == 0


@pytest.mark.asyncio
async def test_initialize_mcp_tools_simple_cases():
    """Test initializing MCP tools from configuration for simple cases only."""
    # Create configuration
    config = {
        "mcp_enabled": False,
        "mcp_config_path": "/path/to/config.json",
        "mcp_tools": {"server1": ["tool1", "tool2"]},
        "provider": "test",
    }

    # Create registry
    mcp_registry = ToolRegistry()

    # Test with mcp_enabled=False
    success, manager = await initialize_mcp_tools(config, mcp_registry, None)
    assert success is False
    assert manager is None

    # Test with missing config_path
    config["mcp_enabled"] = True
    config["mcp_config_path"] = None
    success, manager = await initialize_mcp_tools(config, mcp_registry, None)
    assert success is False
    assert manager is None
