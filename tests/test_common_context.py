"""Tests for the runtime context utilities in common/context.py.

This file contains tests for the centralized context management tools
for runtime context consolidation.
"""

import pytest
import asyncio
from llmproc.common.context import RuntimeContext, validate_context_has
from llmproc.common.metadata import get_tool_meta
from llmproc.tools.core import Tool


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
    context["extra"] = "value"
    assert context["extra"] == "value"

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
        "extra_key": "mock_extra_key",
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
    valid, error = validate_context_has(basic_context, "extra_key")
    assert not valid
    assert "extra_key" in error.lower()

    # Test multiple keys - all present
    valid, error = validate_context_has(full_context, "process", "extra_key")
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

    tool = Tool.from_callable(test_function)
    result = await tool.execute(
        {"param": "test_param"},
        runtime_context={"process": "mock_process"},
    )
    assert result.content == "Got context with process: mock_process"

    result = await tool.execute({"param": "test_param"})
    assert result.is_error
    assert "requires runtime context" in result.content.lower()



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

    tool = Tool.from_callable(test_function)
    result = await tool.execute({"param": "test_param"}, runtime_context={"process": "mock_process"})
    assert result.content == "Got context with process: mock_process"

    result = await tool.execute({"param": "test_param"})
    assert result.is_error
    assert "requires runtime context" in result.content.lower()
