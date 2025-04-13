"""Tests for registry helper functions."""

import pytest

from llmproc.common.results import ToolResult
from llmproc.tools.registry_helpers import (
    apply_aliases_to_schemas,
    check_for_duplicate_schema_names,
    copy_tool_from_source_to_target,
    extract_tool_components,
)
from llmproc.tools.tool_registry import ToolRegistry


async def dummy_handler(args):
    """Simple dummy handler for testing."""
    return ToolResult.from_success("Test result")


def test_extract_tool_components():
    """Test extracting tool components from a registry."""
    registry = ToolRegistry()

    # Register a test tool
    test_schema = {
        "name": "test_tool",
        "description": "A test tool",
        "input_schema": {"type": "object", "properties": {}},
    }
    registry.register_tool("test_tool", dummy_handler, test_schema)

    # Extract components
    success, handler, definition = extract_tool_components(registry, "test_tool")

    # Verify results
    assert success is True
    assert handler is dummy_handler
    assert isinstance(definition, dict)
    assert definition["name"] == "test_tool"
    assert definition["description"] == "A test tool"
    assert definition is not test_schema  # Should be a copy

    # Test with non-existent tool
    success, handler, definition = extract_tool_components(registry, "nonexistent")
    assert success is False
    assert handler is None
    assert definition is None

    # Test with tool that has handler but no definition
    # This is an edge case that shouldn't happen in practice, but we test it anyway
    bad_registry = ToolRegistry()
    bad_registry.tool_handlers["bad_tool"] = dummy_handler
    success, handler, definition = extract_tool_components(bad_registry, "bad_tool")
    assert success is False
    assert (
        handler is not None
    )  # Handler should be returned even if definition not found
    assert definition is None


def test_copy_tool_from_source_to_target():
    """Test copying a tool from one registry to another."""
    source = ToolRegistry()
    target = ToolRegistry()

    # Register a test tool in the source
    test_schema = {
        "name": "test_tool",
        "description": "A test tool",
        "input_schema": {"type": "object", "properties": {}},
    }
    source.register_tool("test_tool", dummy_handler, test_schema)

    # Copy to target
    result = copy_tool_from_source_to_target(source, target, "test_tool")

    # Verify results
    assert result is True
    assert "test_tool" in target.tool_handlers
    assert target.tool_handlers["test_tool"] is dummy_handler

    # Check that the schema was copied
    target_schemas = target.get_definitions()
    assert len(target_schemas) == 1
    assert target_schemas[0]["name"] == "test_tool"

    # Test with non-existent tool
    result = copy_tool_from_source_to_target(source, target, "nonexistent")
    assert result is False


def test_check_for_duplicate_schema_names():
    """Test filtering duplicate schema names."""
    # Create test schemas with duplicates
    schemas = [
        {"name": "tool1", "description": "First tool1"},
        {"name": "tool2", "description": "Tool2"},
        {"name": "tool1", "description": "Second tool1"},
        {"name": "tool3", "description": "Tool3"},
        {"name": "tool2", "description": "Another tool2"},
    ]

    # Filter duplicates
    unique_schemas = check_for_duplicate_schema_names(schemas)

    # Verify results
    assert len(unique_schemas) == 3
    assert unique_schemas[0]["name"] == "tool1"
    assert unique_schemas[0]["description"] == "First tool1"  # First occurrence kept
    assert unique_schemas[1]["name"] == "tool2"
    assert unique_schemas[1]["description"] == "Tool2"  # First occurrence kept
    assert unique_schemas[2]["name"] == "tool3"

    # Test with no duplicates
    no_duplicates = [
        {"name": "tool1", "description": "Tool1"},
        {"name": "tool2", "description": "Tool2"},
        {"name": "tool3", "description": "Tool3"},
    ]
    result = check_for_duplicate_schema_names(no_duplicates)
    assert len(result) == 3
    assert result == no_duplicates

    # Test with empty list
    assert check_for_duplicate_schema_names([]) == []


def test_apply_aliases_to_schemas():
    """Test applying aliases to schemas."""
    # Create test schemas
    schemas = [
        {"name": "original1", "description": "Tool 1"},
        {"name": "original2", "description": "Tool 2"},
        {"name": "original3", "description": "Tool 3"},
    ]

    # Define reverse aliases (original -> alias)
    reverse_aliases = {
        "original1": "alias1",
        "original3": "alias3",
    }

    # Apply aliases
    result = apply_aliases_to_schemas(schemas, reverse_aliases)

    # Verify results
    assert len(result) == 3
    assert result[0]["name"] == "alias1"
    assert result[0]["description"] == "Tool 1"
    assert result[1]["name"] == "original2"  # No alias for this
    assert result[2]["name"] == "alias3"

    # Test with empty aliases
    result = apply_aliases_to_schemas(schemas, {})
    assert len(result) == 3
    assert result[0]["name"] == "original1"
    assert result[1]["name"] == "original2"
    assert result[2]["name"] == "original3"

    # Test with empty schemas
    assert apply_aliases_to_schemas([], reverse_aliases) == []
