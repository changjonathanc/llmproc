"""Tests for the ToolManager class."""

import asyncio
from unittest.mock import Mock, patch

import pytest
from llmproc.common.results import ToolResult
from llmproc.tools import ToolManager, ToolRegistry
from llmproc.tools.builtin import calculator, fd_to_file_tool, fork_tool, read_fd_tool, read_file, spawn_tool
from llmproc.tools.function_tools import register_tool


def test_tool_manager_initialization():
    """Test that ToolManager initializes correctly."""
    manager = ToolManager()

    # Check that the manager has the expected attributes
    assert isinstance(manager.runtime_registry, ToolRegistry)
    assert isinstance(manager.function_tools, list)
    assert len(manager.function_tools) == 0
    # Check registered_tools through getter method
    assert isinstance(manager.get_registered_tools(), list)
    assert len(manager.get_registered_tools()) == 0


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


def test_get_tool_schemas():
    """Test getting tool schemas from the manager by verifying actual schema content."""
    manager = ToolManager()

    # Register a real tool with a specific schema
    calculator_schema = {
        "name": "calculator",
        "description": "Evaluate mathematical expressions",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    }

    # Use a simple mock handler for this test
    async def mock_handler(args):
        return ToolResult.from_success("Result")

    # Register the tool in the runtime registry
    manager.runtime_registry.register_tool("calculator", mock_handler, calculator_schema)

    # Register the calculator tool using the function reference
    from llmproc.tools.builtin import calculator

    manager.register_tools([calculator])

    # Get the schemas
    schemas = manager.get_tool_schemas()

    # Verify schema structure and content
    assert isinstance(schemas, list)
    assert len(schemas) > 0

    # Find our calculator tool schema
    calculator_schema_result = None
    for schema in schemas:
        if schema.get("name") == "calculator":
            calculator_schema_result = schema
            break

    # Verify the specific schema was found and has expected properties
    assert calculator_schema_result is not None
    assert calculator_schema_result.get("description") == "Evaluate mathematical expressions"
    assert "input_schema" in calculator_schema_result
    assert "properties" in calculator_schema_result["input_schema"]
    assert "expression" in calculator_schema_result["input_schema"]["properties"]


@pytest.mark.asyncio
async def test_call_tool():
    """Test calling a tool through the manager."""
    manager = ToolManager()

    # Create a simple mock tool for testing
    async def simple_calculator(expression=None, **kwargs):
        try:
            if not expression:
                return ToolResult.from_error("Missing expression parameter")

            # Simple eval-based calculator (safe for testing only)
            result = eval(expression, {"__builtins__": {}}, {"abs": abs, "max": max, "min": min})
            return ToolResult.from_success(str(result))
        except Exception as e:
            return ToolResult.from_error(f"Error: {str(e)}")

    # Register the calculator tool directly in the runtime registry
    calculator_schema = {
        "name": "calculator",
        "description": "Evaluate mathematical expressions",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    }

    manager.runtime_registry.register_tool("calculator", simple_calculator, calculator_schema)

    # Register the tool using function reference
    from llmproc.tools.builtin import calculator

    manager.register_tools([calculator])

    # Test calling a real tool with real functionality
    result = await manager.call_tool("calculator", {"expression": "3*7+2"})

    # Verify the result
    assert isinstance(result, ToolResult)
    assert result.content == "23"
    assert not result.is_error

    # Test with tool not found to check error handling
    missing_tool_result = await manager.call_tool("missing_tool", {})
    assert missing_tool_result.is_error
    assert missing_tool_result.content == "This tool is not available"

    # Test with a tool registered directly in the registry (now always available)
    manager.runtime_registry.register_tool("disabled_tool", simple_calculator, calculator_schema)
    disabled_result = await manager.call_tool("disabled_tool", {"expression": "1+1"})
    assert not disabled_result.is_error
    assert disabled_result.content == "2"

    # Test error handling for invalid arguments - checking just that the error is set
    result = await manager.call_tool("calculator", {"wrong_arg": "value"})
    assert result.is_error
    # The specific error message could come directly from the tool if it returns a ToolResult
    # or be wrapped by our error handler, so we just check that it exists


@pytest.mark.asyncio
async def test_process_function_tools():
    """Test processing function tools and verifying their behavior."""
    manager = ToolManager()

    # Define a test function with the register_tool decorator
    @register_tool(description="Test doubling function")
    async def double_value(x: int) -> int:
        """Return double the input value.

        Args:
            x: The input value

        Returns:
            The doubled value
        """
        return x * 2

    # Add the function tool
    manager.add_function_tool(double_value)

    # Register the tool using function reference
    manager.register_tools([double_value])

    # Process the function tools (with real processing, not mocked)
    result = manager.process_function_tools()

    # Check the result is the manager itself (for chaining)
    assert result is manager

    # Verify the function tool is registered by using the getter method
    assert "double_value" in manager.get_registered_tools()

    # Verify the tool is registered in the runtime registry
    assert "double_value" in manager.runtime_registry.tool_handlers

    # Most importantly: test that the tool actually works - with explicit parameters
    handler = manager.runtime_registry.get_handler("double_value")
    tool_result = await handler(x=5)
    assert isinstance(tool_result, ToolResult)
    assert not tool_result.is_error
    assert tool_result.content == 10

    # Test with invalid input to verify error handling
    error_result = await handler(wrong_param="value")
    assert error_result.is_error

    # Define another function with a custom name
    @register_tool(name="custom_adder", description="Addition function")
    async def add_numbers(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of the numbers
        """
        return a + b

    # Add and process the second function
    manager.add_function_tool(add_numbers)

    # Register both tools using function references
    manager.register_tools([double_value, add_numbers])

    # Process tools
    manager.process_function_tools()

    # Verify custom name is registered
    assert "custom_adder" in manager.get_registered_tools()
    adder_handler = manager.runtime_registry.get_handler("custom_adder")
    add_result = await adder_handler(a=7, b=3)
    assert add_result.content == 10


@pytest.mark.asyncio
async def test_initialize_tools_directly():
    """Test initializing tools directly through initialize_tools method.

    We test the direct tool initialization through initialize_tools.
    """
    manager = ToolManager()

    # Register the tools in the manager directly using function references
    manager.register_tools([calculator, read_file, fork_tool, spawn_tool, read_fd_tool, fd_to_file_tool])

    # Add function tools to manager
    for tool in [calculator, read_file, fork_tool, spawn_tool, read_fd_tool, fd_to_file_tool]:
        manager.add_function_tool(tool)

    # Create a mock process with properly mocked attributes
    mock_process = Mock()
    mock_process.has_linked_programs = True
    mock_process.linked_programs = {"test_program": Mock()}
    mock_process.linked_program_descriptions = {"test_program": "Test program description"}
    mock_process.file_descriptor_enabled = True

    # Mock the fd_manager to avoid AttributeError
    mock_fd_manager = Mock()
    mock_process.fd_manager = mock_fd_manager

    # Set up the config dictionary with required fields
    config = {
        "fd_manager": mock_fd_manager,
        "linked_programs": mock_process.linked_programs,
        "linked_program_descriptions": mock_process.linked_program_descriptions,
        "has_linked_programs": mock_process.has_linked_programs,
        "provider": "test",
    }

    # Initialize tools directly through initialize_tools
    await manager.initialize_tools(config)

    # Verify tools were registered by checking the runtime registry directly
    assert len(manager.runtime_registry.tool_handlers) >= 6

    # Verify expected tools are in the runtime registry
    expected_tools = [
        "calculator",
        "read_file",
        "fork",
        "spawn",
        "read_fd",
        "fd_to_file",
    ]
    for tool_name in expected_tools:
        assert tool_name in manager.runtime_registry.tool_handlers

    # Test calculator tool by calling it
    calculator_handler = manager.runtime_registry.get_handler("calculator")
    calculator_result = await calculator_handler(expression="2+2")
    assert isinstance(calculator_result, ToolResult)
    assert calculator_result.content == "4"
    assert not calculator_result.is_error

    # Test spawn tool (will return error in mock environment, but should be callable)
    # We need a proper mock context with a process that has linked_programs
    mock_process = {"linked_programs": {"test_program": "mock_program"}}
    runtime_context = {"process": mock_process}
    # Runtime context must be included in the args dictionary
    args = {"program_name": "test_program", "prompt": "Test", "runtime_context": runtime_context}
    spawn_result = await manager.runtime_registry.call_tool("spawn", args)
    assert isinstance(spawn_result, ToolResult)
    # The mock process doesn't actually have linked_programs as a proper attribute
    # so the tool returns success but with an error message in the content
    assert "linked_programs" in str(spawn_result.content)

    # Test that fork tool is registered but returns expected error for direct calls
    # Add runtime_context similarly to spawn tool
    fork_result = await manager.runtime_registry.call_tool(
        "fork", {"prompts": ["Test"], "runtime_context": {"process": "mock_process"}}
    )
    assert isinstance(fork_result, ToolResult)
    # Check that an appropriate error is surfaced (exact wording may evolve)
    assert "Runtime context" in str(fork_result.content) or "Direct calls to fork_tool" in str(fork_result.content)


def test_register_tools_with_mixed_input():
    """Test registering tools from mixed config of strings and callables."""

    # Define a test function
    def test_func(x: int, y: int = 0) -> int:
        """Test function docstring."""
        return x + y

    # Create a tool manager
    manager = ToolManager()

    # Test with mixed input - using function references for builtin tools
    manager.register_tools([calculator, test_func, read_file])

    # Check that the function was added to function_tools
    # With the updated implementation, we expect all 3 functions to be in function_tools
    assert len(manager.function_tools) == 3
    assert test_func in manager.function_tools
    assert calculator in manager.function_tools
    assert read_file in manager.function_tools

    # Test with invalid item type - should raise ValueError
    with pytest.raises(ValueError):
        manager.register_tools([calculator, 123, read_file])

    # Test that valid tools still produce function_tools entries without error
    manager.register_tools([calculator, read_file])
    assert calculator in manager.function_tools
    assert read_file in manager.function_tools


def test_tool_registry_immutability():
    """Test that ToolRegistry methods return copies to prevent external modification."""
    # Create a registry
    registry = ToolRegistry()

    # Create a mock handler
    async def mock_handler(args):
        return ToolResult.from_success("Mock result")

    # Register a few tools
    registry.register_tool("tool1", mock_handler, {"name": "tool1", "description": "Tool 1"})
    registry.register_tool("tool2", mock_handler, {"name": "tool2", "description": "Tool 2"})

    # Test get_definitions returns a copy
    definitions = registry.get_definitions()

    # Try to modify the returned list
    definitions.append({"name": "tool3", "description": "Should not be added"})

    # Check the original list in the registry is unchanged
    registry_defs = registry.get_definitions()
    assert len(registry_defs) == 2
    assert all(d["name"] in ["tool1", "tool2"] for d in registry_defs)

    # Test that get_tool_names returns a copy
    tool_names = registry.get_tool_names()
    tool_names.append("should_not_be_added")

    # Check the original mapping is unchanged
    assert "should_not_be_added" not in registry.tool_handlers
