"""Tests for the context-aware tool decorator and runtime context injection."""

import asyncio

import pytest

from llmproc.common.results import ToolResult
from llmproc.tools.context_aware import context_aware, is_context_aware
from llmproc.tools.tool_manager import ToolManager


# Define test handler functions
async def standard_handler(param):
    """A standard handler that doesn't need context."""
    return ToolResult.from_success(f"Standard handler called with: {param}")


@context_aware
async def context_aware_handler(param, runtime_context=None):
    """A context-aware handler that uses runtime context."""
    if not runtime_context or "test_value" not in runtime_context:
        return ToolResult.from_error("Missing or invalid runtime context")

    test_value = runtime_context.get("test_value")
    return ToolResult.from_success(
        f"Context-aware handler called with: {param}, context: {test_value}"
    )


class TestContextAware:
    """Tests for the context-aware decorator and runtime context injection."""

    def test_is_context_aware_detection(self):
        """Test that is_context_aware correctly identifies decorated functions."""
        # Standard handler should not be marked as context-aware
        assert not is_context_aware(standard_handler)

        # Context-aware handler should be marked as context-aware
        assert is_context_aware(context_aware_handler)

    def test_context_aware_decorator_preserves_function_name(self):
        """Test that the decorator preserves the function name and docstring."""
        assert context_aware_handler.__name__ == "context_aware_handler"
        assert "context-aware handler" in context_aware_handler.__doc__

    @pytest.mark.asyncio
    async def test_tool_manager_context_injection(self):
        """Test that ToolManager injects runtime context for context-aware tools."""
        # Create a tool manager
        manager = ToolManager()

        # Set up runtime context
        manager.set_runtime_context({"test_value": "Hello from context!"})

        # Register both standard and context-aware tools
        standard_def = {
            "name": "standard_tool",
            "description": "A standard tool",
            "input_schema": {"type": "object", "properties": {}},
        }

        context_def = {
            "name": "context_tool",
            "description": "A context-aware tool",
            "input_schema": {"type": "object", "properties": {}},
        }

        manager.runtime_registry.register_tool(
            "standard_tool", standard_handler, standard_def
        )
        manager.runtime_registry.register_tool(
            "context_tool", context_aware_handler, context_def
        )

        # Add both tools to enabled_tools
        manager.enabled_tools = ["standard_tool", "context_tool"]

        # Call both tools
        standard_result = await manager.call_tool("standard_tool", {"param": "value"})
        context_result = await manager.call_tool("context_tool", {"param": "value"})

        # Verify results
        assert not standard_result.is_error
        assert "Standard handler called with" in standard_result.content

        assert not context_result.is_error
        assert "Context-aware handler called with" in context_result.content
        assert "Hello from context!" in context_result.content

    @pytest.mark.asyncio
    async def test_missing_context_handling(self):
        """Test that context-aware tools handle missing context gracefully."""
        # Create a tool manager without setting runtime context
        manager = ToolManager()

        # Register a context-aware tool
        context_def = {
            "name": "context_tool",
            "description": "A context-aware tool",
            "input_schema": {"type": "object", "properties": {}},
        }

        manager.runtime_registry.register_tool(
            "context_tool", context_aware_handler, context_def
        )
        manager.enabled_tools = ["context_tool"]

        # Call the tool without setting runtime context first
        result = await manager.call_tool("context_tool", {"param": "value"})

        # Verify result is an error due to missing context
        assert result.is_error
        assert "Missing or invalid runtime context" in result.content
