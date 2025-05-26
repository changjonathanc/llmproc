"""Tests for the tool registration callback functionality."""

from unittest.mock import Mock

import pytest
from llmproc.common.metadata import ToolMeta, get_tool_meta
from llmproc.tools.function_tools import register_tool
from llmproc.tools.tool_manager import ToolManager


def test_on_register_callback_in_metadata():
    """Test that on_register callback is stored in tool metadata."""

    # Define a simple callback function
    def my_callback(tool_name, tool_manager):
        pass

    # Register a tool with on_register callback
    @register_tool(name="test_tool", description="A test tool", on_register=my_callback)
    async def test_tool(arg: str):
        """A test tool function."""
        return f"Processed: {arg}"

    # Check that the callback is stored in metadata
    meta = get_tool_meta(test_tool)
    assert meta.on_register is not None
    assert meta.on_register is my_callback


def test_callback_execution_during_registration():
    """Test that on_register callback is executed during tool registration."""
    # Create a mock callback
    mock_callback = Mock()

    # Register a tool with the mock callback
    @register_tool(name="callback_test", description="Test callback execution", on_register=mock_callback)
    async def callback_test(arg: str):
        """A test tool function."""
        return f"Result: {arg}"

    # Create a tool manager and register the tool
    manager = ToolManager()
    manager.add_function_tool(callback_test)
    manager.process_function_tools()

    # Check that the callback was called with the correct arguments
    mock_callback.assert_called_once()
    args, kwargs = mock_callback.call_args
    assert args[0] == "callback_test"  # tool_name
    assert args[1] is manager  # tool_manager


def test_callback_sets_attributes():
    """Test that on_register callback can set attributes on the tool manager."""

    # Define a callback that sets an attribute
    def set_custom_attr(tool_name, tool_manager):
        tool_manager.custom_attribute = f"Set by {tool_name}"

    # Register a tool with the callback
    @register_tool(name="attr_setter", description="Tool that sets attribute", on_register=set_custom_attr)
    async def attr_setter():
        """A test tool function."""
        return "Done"

    # Create a tool manager and register the tool
    manager = ToolManager()
    manager.add_function_tool(attr_setter)
    manager.process_function_tools()

    # Check that the attribute was set
    assert hasattr(manager, "custom_attribute")
    assert manager.custom_attribute == "Set by attr_setter"


def test_goto_tool_enables_message_ids():
    """Test that the goto tool's on_register callback enables message IDs."""
    # Import the goto tool
    from llmproc.tools.builtin.goto import handle_goto

    # Create a tool manager and register the goto tool
    manager = ToolManager()
    manager.add_function_tool(handle_goto)
    manager.process_function_tools()

    # Check that message_ids_enabled attribute was set
    assert hasattr(manager, "message_ids_enabled")
    assert manager.message_ids_enabled is True


def test_callback_error_handling():
    """Test that errors in on_register callbacks are handled properly."""

    # Define a callback that raises an exception
    def error_callback(tool_name, tool_manager):
        raise ValueError("Test error")

    # Register a tool with the error callback
    @register_tool(name="error_tool", description="Tool with error callback", on_register=error_callback)
    async def error_tool():
        """A test tool function."""
        return "Done"

    # Create a tool manager and register the tool
    manager = ToolManager()
    manager.add_function_tool(error_tool)

    # The process_function_tools method should not raise an exception
    manager.process_function_tools()  # Should not raise

    # Tool should still be registered despite callback error
    assert "error_tool" in manager.registered_tools
