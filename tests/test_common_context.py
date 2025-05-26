"""Tests for the runtime context utilities in common/context.py.

This file contains tests for the centralized context management tools
for runtime context consolidation.
"""

import pytest
from llmproc.common.context import RuntimeContext, validate_context_has
from llmproc.common.metadata import get_tool_meta


# Local helper replicating former API for concise tests
def check_requires_context(handler):  # noqa: D401
    """Return True if the handler requires runtime context."""
    return get_tool_meta(handler).requires_context


from llmproc.common.results import ToolResult
from llmproc.tools.function_tools import register_tool
from llmproc.tools.tool_manager import ToolManager


def test_runtime_context_type():
    """Test that the RuntimeContext type can be instantiated and used."""
    # Create a minimal context
    context: RuntimeContext = {"process": "mock_process"}

    # Should allow direct key access
    assert context["process"] == "mock_process"

    # Should allow adding new keys
    context["fd_manager"] = "mock_fd_manager"
    assert context["fd_manager"] == "mock_fd_manager"

    # Should work with get method
    assert context.get("process") == "mock_process"
    assert context.get("nonexistent") is None


def test_validate_context_has():
    """Test validation of context keys."""
    # Create test contexts
    empty_context = {}
    basic_context = {"process": "mock_process"}
    full_context = {
        "process": "mock_process",
        "fd_manager": "mock_fd_manager",
        "linked_programs": {"test": "program"},
    }

    # Test with None
    valid, error = validate_context_has(None)
    assert not valid
    assert "missing" in error.lower()

    # Test empty context with no keys
    valid, error = validate_context_has({})
    assert valid, "Empty context with no required keys should be valid"
    assert error is None, f"Error should be None, got: {error}"

    # Test empty context with required key
    valid, error = validate_context_has(empty_context, "process")
    assert not valid
    assert "missing required keys: process" in error.lower()

    # Test with basic context
    valid, error = validate_context_has(basic_context, "process")
    assert valid
    assert error is None

    # Test missing key
    valid, error = validate_context_has(basic_context, "fd_manager")
    assert not valid
    assert "fd_manager" in error.lower()

    # Test multiple keys - all present
    valid, error = validate_context_has(full_context, "process", "fd_manager")
    assert valid
    assert error is None

    # Test multiple keys - some missing
    valid, error = validate_context_has(full_context, "process", "missing_key")
    assert not valid
    assert "missing_key" in error.lower()


def test_register_tool_with_requires_context_simple():
    """Test the register_tool decorator with requires_context=True."""

    # Define a context-aware function (using register_tool)
    @register_tool(requires_context=True)
    async def test_function(param, runtime_context=None):
        if runtime_context and "process" in runtime_context:
            return f"Got context with process: {runtime_context['process']}"
        return "No context or missing process"

    # Verify it's marked as context-aware using metadata system
    from llmproc.common.metadata import get_tool_meta

    meta = get_tool_meta(test_function)
    assert meta.requires_context is True
    assert check_requires_context(test_function)

    # Test detecting non-context-aware function
    async def regular_function(param):
        return param

    assert not check_requires_context(regular_function)


@pytest.mark.asyncio
async def test_requires_context_execution():
    """Test execution of function with requires_context=True with and without context."""

    # Define a context-aware function (using register_tool)
    @register_tool(requires_context=True)
    async def test_function(param, runtime_context=None):
        if runtime_context and "process" in runtime_context:
            return f"Got context with process: {runtime_context['process']}"
        return "No context or missing process"

    # Test with context
    result = await test_function("test_param", runtime_context={"process": "mock_process"})
    assert result == "Got context with process: mock_process"

    # Test without context
    result = await test_function("test_param")
    assert result == "No context or missing process"


@pytest.mark.asyncio
async def test_register_tool_with_required_context_keys():
    """Test register_tool with required_context_keys parameter."""

    # Define a function with required context keys (using register_tool)
    @register_tool(requires_context=True, required_context_keys=["process", "fd_manager"])
    async def test_function(param, runtime_context=None):
        # Add null check to prevent error when decorator doesn't provide context
        if not runtime_context:
            return "Missing context"
        return f"Got process: {runtime_context['process']} and fd: {runtime_context['fd_manager']}"

    # Verify required keys are stored in metadata
    from llmproc.common.metadata import get_tool_meta

    meta = get_tool_meta(test_function)
    assert meta.required_context_keys
    assert "process" in meta.required_context_keys
    assert "fd_manager" in meta.required_context_keys

    # Test with complete context
    complete_context = {"process": "mock_process", "fd_manager": "mock_fd_manager"}
    result = await test_function("test_param", runtime_context=complete_context)
    assert "Got process: mock_process and fd: mock_fd_manager" == result

    # Test with incomplete context
    incomplete_context = {"process": "mock_process"}
    result = await test_function("test_param", runtime_context=incomplete_context)
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "missing required keys" in result.content.lower()
    assert "fd_manager" in result.content

    # Test with missing context
    result = await test_function("test_param")
    # The function should return "Missing context" if runtime_context is None
    # This is our test implementation's null check, not an error
    assert result == "Missing context"


def test_check_requires_context_function():
    """Test the check_requires_context utility function."""

    # Test with context-aware function (using register_tool)
    @register_tool(requires_context=True)
    async def context_function(param, runtime_context=None):
        return param

    assert check_requires_context(context_function)

    # Test with register_tool(requires_context=True)
    @register_tool(requires_context=True)
    async def register_context_function(param, runtime_context=None):
        return param

    assert check_requires_context(register_context_function)

    # Test with non-context-aware function
    async def regular_function(param):
        return param

    assert not check_requires_context(regular_function)

    # Test with normal object
    obj = object()
    assert not check_requires_context(obj)


@pytest.mark.asyncio
async def test_register_tool_with_required_context():
    """Test the register_tool decorator with requires_context=True."""

    # Define a function with register_tool
    @register_tool(requires_context=True)
    async def test_function(param, runtime_context=None):
        if runtime_context and "process" in runtime_context:
            return f"Got context with process: {runtime_context['process']}"
        return "No context or missing process"

    # Verify it's marked as context-aware using metadata
    from llmproc.common.metadata import get_tool_meta

    meta = get_tool_meta(test_function)
    assert meta.requires_context is True
    assert check_requires_context(test_function)

    # Test with context
    result = await test_function("test_param", runtime_context={"process": "mock_process"})
    assert result == "Got context with process: mock_process"

    # Test without context
    result = await test_function("test_param")
    assert result == "No context or missing process"


@pytest.mark.asyncio
async def test_register_tool_with_required_keys():
    """Test register_tool with required_context_keys."""

    # Define a function with register_tool and required_context_keys
    @register_tool(requires_context=True, required_context_keys=["process", "fd_manager"])
    async def test_function(param, runtime_context=None):
        if not runtime_context:
            return "Missing context"
        return f"Got process: {runtime_context['process']} and fd: {runtime_context['fd_manager']}"

    # Verify required keys are stored in metadata
    from llmproc.common.metadata import get_tool_meta

    meta = get_tool_meta(test_function)
    assert meta.required_context_keys
    assert "process" in meta.required_context_keys
    assert "fd_manager" in meta.required_context_keys

    # Test with complete context
    complete_context = {"process": "mock_process", "fd_manager": "mock_fd_manager"}
    result = await test_function("test_param", runtime_context=complete_context)
    assert "Got process: mock_process and fd: mock_fd_manager" == result

    # Test with incomplete context
    incomplete_context = {"process": "mock_process"}
    result = await test_function("test_param", runtime_context=incomplete_context)
    # With register_tool(requires_context=True, required_context_keys=["process", "fd_manager"]),
    # the decorator will automatically validate and return a ToolResult error
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "missing required keys: fd_manager" in result.content.lower()


@pytest.mark.asyncio
async def test_register_tool_with_tool_manager():
    """Test register_tool with ToolManager integration."""

    # Define a function with register_tool
    @register_tool(
        name="test_tool",
        description="Test tool with context requirement",
        requires_context=True,
        required_context_keys=["process", "fd_manager"],
    )
    async def test_function(param, runtime_context=None):
        if not runtime_context:
            return ToolResult.from_error("Missing context")
        if "process" not in runtime_context or "fd_manager" not in runtime_context:
            return ToolResult.from_error("Missing required keys")
        return ToolResult.from_success(
            f"Got process: {runtime_context['process']} and fd: {runtime_context['fd_manager']}"
        )

    # Create a tool manager with context
    manager = ToolManager()
    manager.set_runtime_context({"process": "mock_process", "fd_manager": "mock_fd_manager"})

    # Register and enable the tool
    schema = {"name": "test_tool", "description": "Test tool", "input_schema": {"type": "object", "properties": {}}}
    manager.runtime_registry.register_tool("test_tool", test_function, schema)
    manager.register_tools([test_function])

    # Call through the manager
    result = await manager.call_tool("test_tool", {"param": "value"})

    # Verify we got the context
    assert isinstance(result, ToolResult)
    assert not result.is_error
    assert "Got process: mock_process" in result.content
