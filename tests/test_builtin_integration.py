"""Tests for builtin tool integration functions.

These tests verify the integration functions in the builtin.integration module.
"""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from llmproc.common.results import ToolResult
from llmproc.tools.builtin.integration import (
    copy_tool_from_source_to_target,
    load_builtin_tools,
    register_fd_tool,
    register_fork_tool,
    register_goto_tool,
    register_spawn_tool,
    register_system_tools,
)
from llmproc.tools.tool_registry import ToolRegistry


def test_load_builtin_tools():
    """Test loading builtin tools into a registry."""
    # Create a mock registry
    registry = ToolRegistry()

    # Load builtin tools into the registry
    result = load_builtin_tools(registry)

    # Verify the result is True
    assert result is True

    # Verify that the registry was populated
    assert len(registry.tool_handlers) > 0

    # Check for specific tools
    assert "fork" in registry.tool_handlers
    assert "spawn" in registry.tool_handlers
    assert "goto" in registry.tool_handlers
    assert "read_fd" in registry.tool_handlers
    assert "fd_to_file" in registry.tool_handlers
    assert "calculator" in registry.tool_handlers
    assert "read_file" in registry.tool_handlers
    assert "list_dir" in registry.tool_handlers

    # Check that tool definitions were properly registered
    assert any(d.get("name") == "fork" for d in registry.get_definitions())
    assert any(d.get("name") == "calculator" for d in registry.get_definitions())


def test_copy_tool_from_source_to_target():
    """Test copying a tool from source registry to target registry."""
    # Create source and target registries
    source_registry = ToolRegistry()
    target_registry = ToolRegistry()

    # Register a test tool in the source registry
    async def test_handler(args):
        return ToolResult.from_success("Test result")

    test_schema = {
        "name": "test_tool",
        "description": "Test tool",
        "input_schema": {"type": "object", "properties": {}},
    }

    source_registry.register_tool("test_tool", test_handler, test_schema)

    # Copy the tool from source to target
    result = copy_tool_from_source_to_target(
        source_registry, target_registry, "test_tool"
    )

    # Verify the result is True
    assert result is True

    # Verify that the tool was copied
    assert "test_tool" in target_registry.tool_handlers

    # Verify that the tool definition was copied
    assert any(d.get("name") == "test_tool" for d in target_registry.get_definitions())


def test_register_fork_tool():
    """Test registering the fork tool."""
    # Create source and target registries
    source_registry = ToolRegistry()
    target_registry = ToolRegistry()

    # Register a test tool in the source registry
    async def test_handler(args):
        return ToolResult.from_success("Test result")

    test_schema = {
        "name": "fork",
        "description": "Fork tool",
        "input_schema": {"type": "object", "properties": {}},
    }

    source_registry.register_tool("fork", test_handler, test_schema)

    # Register the fork tool
    result = register_fork_tool(source_registry, target_registry, "fork")

    # Verify the result is True
    assert result is True

    # Verify that the tool was registered
    assert "fork" in target_registry.tool_handlers


def test_register_goto_tool():
    """Test registering the goto tool."""
    # Create source and target registries
    source_registry = ToolRegistry()
    target_registry = ToolRegistry()

    # Register a test tool in the source registry
    async def test_handler(args):
        return ToolResult.from_success("Test result")

    test_schema = {
        "name": "goto",
        "description": "Goto tool",
        "input_schema": {"type": "object", "properties": {}},
    }

    source_registry.register_tool("goto", test_handler, test_schema)

    # Register the goto tool
    result = register_goto_tool(source_registry, target_registry, "goto")

    # Verify the result is True
    assert result is True

    # Verify that the tool was registered
    assert "goto" in target_registry.tool_handlers


def test_register_spawn_tool():
    """Test registering the spawn tool."""
    # Create source and target registries
    source_registry = ToolRegistry()
    target_registry = ToolRegistry()

    # Register a test tool in the source registry
    async def test_handler(args):
        return ToolResult.from_success("Test result")

    test_schema = {
        "name": "spawn",
        "description": "Spawn tool",
        "input_schema": {"type": "object", "properties": {}},
    }

    source_registry.register_tool("spawn", test_handler, test_schema)

    # Create linked programs
    linked_programs = {"test_program": MagicMock()}
    linked_program_descriptions = {"test_program": "Test program"}

    # Register the spawn tool
    result = register_spawn_tool(
        source_registry,
        target_registry,
        "spawn",
        linked_programs,
        linked_program_descriptions,
    )

    # Verify the result is True
    assert result is True

    # Verify that the tool was registered
    assert "spawn" in target_registry.tool_handlers


def test_register_fd_tool():
    """Test registering an FD tool."""
    # Create source and target registries
    source_registry = ToolRegistry()
    target_registry = ToolRegistry()

    # Register a test tool in the source registry
    async def test_handler(args):
        return ToolResult.from_success("Test result")

    test_schema = {
        "name": "read_fd",
        "description": "Read FD tool",
        "input_schema": {"type": "object", "properties": {}},
    }

    source_registry.register_tool("read_fd", test_handler, test_schema)

    # Create a mock FD manager
    fd_manager = MagicMock()
    fd_manager.register_fd_tool = MagicMock()

    # Register the FD tool
    result = register_fd_tool(source_registry, target_registry, "read_fd", fd_manager)

    # Verify the result is True
    assert result is True

    # Verify that the tool was registered with the FD manager
    fd_manager.register_fd_tool.assert_called_once_with("read_fd")

    # Verify that the tool was registered in the target registry
    assert "read_fd" in target_registry.tool_handlers


@pytest.mark.asyncio
async def test_register_system_tools():
    """Test registering system tools based on configuration."""
    # Create source and target registries
    source_registry = ToolRegistry()
    target_registry = ToolRegistry()

    # Load builtin tools into the source registry
    load_builtin_tools(source_registry)

    # Create a configuration
    config = {
        "fd_manager": MagicMock(),
        "linked_programs": {"test_program": MagicMock()},
        "linked_program_descriptions": {"test_program": "Test program"},
        "has_linked_programs": True,
    }

    # Register system tools
    enabled_tools = ["calculator", "fork", "spawn", "read_fd"]
    registered_count = register_system_tools(
        source_registry, target_registry, enabled_tools, config
    )

    # Verify that the correct number of tools were registered
    assert registered_count == len(enabled_tools)

    # Verify that the tools were registered in the target registry
    for tool_name in enabled_tools:
        assert tool_name in target_registry.tool_handlers


@pytest.mark.asyncio
async def test_register_system_tools_fd_auto_enable():
    """Test that read_fd is automatically enabled when fd_manager is available."""
    # Create source and target registries
    source_registry = ToolRegistry()
    target_registry = ToolRegistry()

    # Load builtin tools into the source registry
    load_builtin_tools(source_registry)

    # Create a configuration with fd_manager
    config = {
        "fd_manager": MagicMock(),
        "linked_programs": {},
        "linked_program_descriptions": {},
        "has_linked_programs": False,
    }

    # Register system tools with empty enabled_tools list
    enabled_tools = []
    registered_count = register_system_tools(
        source_registry, target_registry, enabled_tools, config
    )

    # Verify that read_fd was automatically enabled and registered
    assert registered_count == 1
    assert "read_fd" in target_registry.tool_handlers


@pytest.mark.asyncio
async def test_register_system_tools_spawn_skipped():
    """Test that spawn tool is skipped when has_linked_programs is False."""
    # Create source and target registries
    source_registry = ToolRegistry()
    target_registry = ToolRegistry()

    # Load builtin tools into the source registry
    load_builtin_tools(source_registry)

    # Create a configuration without linked programs
    config = {
        "fd_manager": None,
        "linked_programs": {},
        "linked_program_descriptions": {},
        "has_linked_programs": False,
    }

    # Register system tools with spawn in enabled_tools
    enabled_tools = ["spawn"]
    registered_count = register_system_tools(
        source_registry, target_registry, enabled_tools, config
    )

    # Verify that no tools were registered
    assert registered_count == 0
    assert "spawn" not in target_registry.tool_handlers
