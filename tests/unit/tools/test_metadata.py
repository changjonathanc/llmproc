"""Unit tests for tool metadata system.

These tests verify the functionality of the tool metadata encapsulation system.
"""

import asyncio
from typing import Any

import pytest
from llmproc.common.access_control import AccessLevel
from llmproc.common.constants import TOOL_METADATA_ATTR
from llmproc.common.metadata import ToolMeta, attach_meta, get_tool_meta
from llmproc.tools.function_tools import register_tool


def test_tool_meta_creation():
    """Test creation of ToolMeta data class."""
    meta = ToolMeta(
        name="test_tool", description="Test tool description", access=AccessLevel.READ, requires_context=True
    )

    assert meta.name == "test_tool"
    assert meta.description == "Test tool description"
    assert meta.access == AccessLevel.READ
    assert meta.requires_context is True
    assert meta.required_context_keys == ()


def test_attach_meta_retrieval():
    """Test attaching and retrieving metadata."""

    # Create a simple test function
    def test_func():
        return "test"

    # Create metadata and attach it
    meta = ToolMeta(name="custom_name", access=AccessLevel.ADMIN)
    attach_meta(test_func, meta)

    # Verify the metadata was attached correctly
    assert hasattr(test_func, TOOL_METADATA_ATTR)
    assert getattr(test_func, TOOL_METADATA_ATTR) is meta

    # Verify retrieval with helper
    retrieved = get_tool_meta(test_func)
    assert retrieved is meta
    assert retrieved.name == "custom_name"
    assert retrieved.access == AccessLevel.ADMIN


def test_register_tool_plain_decorator():
    """Test metadata with plain @register_tool decorator."""

    @register_tool
    def plain_tool(arg1: str, arg2: int = 0):
        """A simple tool."""
        return f"{arg1} {arg2}"

    # Check metadata
    meta = get_tool_meta(plain_tool)
    assert meta.name == "plain_tool"  # Default to function name
    assert meta.access == AccessLevel.WRITE  # Default access level
    assert meta.requires_context is False


def test_register_tool_with_params():
    """Test metadata with parameterized @register_tool decorator."""

    @register_tool(
        name="custom_name",
        description="Custom description",
        access=AccessLevel.READ,
        requires_context=True,
        required_context_keys=["process"],
    )
    async def param_tool(arg: str, runtime_context=None):
        """Tool docstring."""
        return arg

    # Check metadata
    meta = get_tool_meta(param_tool)
    assert meta.name == "custom_name"
    assert meta.description == "Custom description"
    assert meta.access == AccessLevel.READ
    assert meta.requires_context is True
    assert "process" in meta.required_context_keys


def test_register_tool_with_string_access():
    """Test metadata with string access level in decorator."""

    @register_tool(access="admin")
    def admin_tool():
        return "admin action"

    # Check metadata
    meta = get_tool_meta(admin_tool)
    assert meta.access == AccessLevel.ADMIN


@pytest.mark.asyncio
async def test_mcp_tool_metadata_simulation():
    """Simulate MCP tool registration and verify metadata."""

    # This simulates how MCP tools are registered
    async def fake_mcp_tool(**kwargs):
        return "mcp result"

    # Attach metadata as done in MCPManager
    meta = ToolMeta(name="server__tool", description="MCP tool description", access=AccessLevel.READ)
    attach_meta(fake_mcp_tool, meta)

    # Verify metadata
    retrieved = get_tool_meta(fake_mcp_tool)
    assert retrieved.name == "server__tool"
    assert retrieved.description == "MCP tool description"
    assert retrieved.access == AccessLevel.READ

    # Verify we can call the tool
    result = await fake_mcp_tool(param="test")
    assert result == "mcp result"


def test_metadata_default_values():
    """Test default values in ToolMeta."""

    # Create a function and don't attach metadata
    def no_meta_func():
        pass

    # Using get_tool_meta should return a default ToolMeta instance
    meta = get_tool_meta(no_meta_func)
    assert isinstance(meta, ToolMeta)
    assert meta.name is None
    assert meta.description is None
    assert meta.access == AccessLevel.WRITE
    assert meta.requires_context is False
    assert meta.required_context_keys == ()


def test_metadata_for_context_wrapper():
    """Test metadata is properly attached to context wrappers."""

    @register_tool(requires_context=True, required_context_keys=["process", "fd_manager"])
    async def context_tool(param: str, runtime_context=None):
        return f"processed {param}"

    # The wrapper should have metadata
    meta = get_tool_meta(context_tool)
    assert meta.requires_context is True
    assert "process" in meta.required_context_keys
    assert "fd_manager" in meta.required_context_keys

    # The name should default to the function name
    assert meta.name == "context_tool"


def test_compare_access_levels():
    """Test the compare_to method in AccessLevel."""
    # Test all combinations
    assert AccessLevel.READ.compare_to(AccessLevel.READ) == 0
    assert AccessLevel.READ.compare_to(AccessLevel.WRITE) < 0
    assert AccessLevel.READ.compare_to(AccessLevel.ADMIN) < 0

    assert AccessLevel.WRITE.compare_to(AccessLevel.READ) > 0
    assert AccessLevel.WRITE.compare_to(AccessLevel.WRITE) == 0
    assert AccessLevel.WRITE.compare_to(AccessLevel.ADMIN) < 0

    assert AccessLevel.ADMIN.compare_to(AccessLevel.READ) > 0
    assert AccessLevel.ADMIN.compare_to(AccessLevel.WRITE) > 0
    assert AccessLevel.ADMIN.compare_to(AccessLevel.ADMIN) == 0
