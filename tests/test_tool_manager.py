"""Tests for the ToolManager class."""

import pytest
import asyncio
from unittest.mock import Mock, patch

from llmproc.tools import ToolManager, ToolNotFoundError, ToolRegistry, ToolResult
from llmproc.tools.function_tools import register_tool


def test_tool_manager_initialization():
    """Test that ToolManager initializes correctly."""
    manager = ToolManager()
    
    # Check that the manager has the expected attributes
    assert isinstance(manager.registry, ToolRegistry)
    assert isinstance(manager.function_tools, list)
    assert isinstance(manager.enabled_tools, list)
    assert len(manager.function_tools) == 0
    assert len(manager.enabled_tools) == 0


def test_add_function_tool():
    """Test adding function tools to the manager."""
    manager = ToolManager()
    
    # Define a test function
    def test_func(x: int) -> int:
        return x * 2
        
    # Add the function
    result = manager.add_function_tool(test_func)
    
    # Check the result is the manager itself (for chaining)
    assert result is manager
    
    # Check the function was added
    assert len(manager.function_tools) == 1
    assert manager.function_tools[0] is test_func
    
    # Test with non-callable
    with pytest.raises(ValueError):
        manager.add_function_tool("not a function")


# test_add_dict_tool test removed as the method has been removed from ToolManager


def test_get_tool_schemas():
    """Test getting tool schemas from the manager."""
    manager = ToolManager()
    
    # Mock the registry's get_definitions method
    manager.registry.get_definitions = Mock(return_value=[{"name": "test_tool"}])
    
    # Get the schemas
    schemas = manager.get_tool_schemas()
    
    # Check the schemas
    assert schemas == [{"name": "test_tool"}]
    
    # Verify the mock was called
    manager.registry.get_definitions.assert_called_once()


@pytest.mark.asyncio
async def test_call_tool():
    """Test calling a tool through the manager."""
    manager = ToolManager()
    
    # Mock the registry's call_tool method
    mock_result = ToolResult(content="result", is_error=False)
    manager.registry.call_tool = Mock(return_value=asyncio.Future())
    manager.registry.call_tool.return_value.set_result(mock_result)
    
    # Call the tool
    result = await manager.call_tool("test_tool", {"arg": "value"})
    
    # Check the result
    assert result is mock_result
    
    # Verify the mock was called
    manager.registry.call_tool.assert_called_once_with("test_tool", {"arg": "value"})
    
    # Test with tool not found
    manager.registry.call_tool = Mock(side_effect=ValueError("Tool 'missing_tool' not found"))
    
    # Call the tool and expect ToolNotFoundError
    with pytest.raises(ToolNotFoundError):
        await manager.call_tool("missing_tool", {})


@pytest.mark.asyncio
async def test_process_function_tools():
    """Test processing function tools."""
    manager = ToolManager()
    
    # Define a test function with the register_tool decorator
    @register_tool(description="Test function")
    def test_func(x: int) -> int:
        """Return double the input.
        
        Args:
            x: The input value
            
        Returns:
            The doubled value
        """
        return x * 2
        
    # Add the function
    manager.add_function_tool(test_func)
    
    # Process the function tools
    with patch('llmproc.tools.tool_manager.create_tool_from_function') as mock_create:
        # Mock the create_tool_from_function function
        mock_handler = Mock()
        mock_schema = {"name": "test_func", "description": "Test function"}
        mock_create.return_value = (mock_handler, mock_schema)
        
        # Mock the register_tool method
        manager.registry.register_tool = Mock()
        
        # Process the function tools
        result = manager.process_function_tools()
        
        # Check the result is the manager itself (for chaining)
        assert result is manager
        
        # Verify the mocks were called
        mock_create.assert_called_once_with(test_func)
        manager.registry.register_tool.assert_called_once_with("test_func", mock_handler, mock_schema)
        
        # Check the tool name was added to enabled_tools
        assert "test_func" in manager.enabled_tools


def test_register_system_tools():
    """Test registering system tools."""
    manager = ToolManager()
    
    # Create a mock process with properly mocked attributes
    mock_process = Mock()
    mock_process.enabled_tools = ["calculator", "read_file", "fork", "spawn", "read_fd", "fd_to_file"]
    mock_process.has_linked_programs = True
    mock_process.linked_programs = {"test_program": Mock()}
    mock_process.linked_program_descriptions = {"test_program": "Test program description"}
    mock_process.file_descriptor_enabled = True
    
    # Mock the fd_manager to avoid AttributeError
    mock_fd_manager = Mock()
    mock_process.fd_manager = mock_fd_manager
    
    # Register system tools
    result = manager.register_system_tools(mock_process)
    
    # Check the result is the manager itself (for chaining)
    assert result is manager
    
    # Verify that tools were registered properly
    assert len(manager.registry.tool_handlers) > 0
    assert "calculator" in manager.tool_handlers
    assert "read_file" in manager.tool_handlers
    assert "fork" in manager.tool_handlers
    assert "spawn" in manager.tool_handlers
    assert "read_fd" in manager.tool_handlers
    assert "fd_to_file" in manager.tool_handlers
    
    # Just verify we have the tools registered properly
    assert len(manager.tool_handlers) >= 6