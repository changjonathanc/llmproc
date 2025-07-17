"""Tests for the ToolManager class."""

import asyncio
from unittest.mock import MagicMock, Mock, patch

import pytest

from llmproc.common.access_control import AccessLevel
from llmproc.common.metadata import get_tool_meta
from llmproc.common.results import ToolResult
from llmproc.plugin.plugin_event_runner import PluginEventRunner
from llmproc.plugins.file_descriptor import FileDescriptorManager, FileDescriptorPlugin
from llmproc.plugins.spawn import SpawnPlugin
from llmproc.tools import ToolManager, ToolRegistry
from llmproc.tools.builtin import calculator, fork_tool, read_file, spawn_tool
from llmproc.tools.core import Tool
from llmproc.tools.function_tools import register_tool


def test_tool_manager_starts_with_empty_registry():
    """ToolManager should start with an empty runtime registry."""
    manager = ToolManager()

    # Check that the manager has the expected attributes
    assert isinstance(manager.runtime_registry, ToolRegistry)
    assert isinstance(manager.registered_tools, list)
    assert len(manager.registered_tools) == 0


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
    manager.runtime_registry.register_tool_obj(
        Tool(
            handler=mock_handler,
            schema=calculator_schema,
            meta=get_tool_meta(mock_handler),
        )
    )

    # Register the calculator tool using the function reference
    from llmproc.tools.builtin import calculator

    asyncio.run(manager.register_tools([calculator], {}))

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

    # Register the primary calculator tool
    manager.runtime_registry.register_tool_obj(
        Tool(
            handler=simple_calculator,
            schema=calculator_schema,
            meta=get_tool_meta(simple_calculator),
        )
    )

    # Also register a variant under a different name
    disabled_schema = calculator_schema.copy()
    disabled_schema["name"] = "disabled_tool"
    manager.runtime_registry.register_tool_obj(
        Tool(
            handler=simple_calculator,
            schema=disabled_schema,
            meta=get_tool_meta(simple_calculator),
        )
    )

    # Register the tool using function reference
    from llmproc.tools.builtin import calculator

    await manager.register_tools([calculator], {})

    # Test calling a real tool with real functionality
    result = await manager.call_tool("calculator", {"expression": "3*7+2"})

    # Verify the result
    assert isinstance(result, ToolResult)
    assert result.content == "23"
    assert not result.is_error

    # Test with tool not found to check error handling
    missing_tool_result = await manager.call_tool("missing_tool", {})
    assert missing_tool_result.is_error
    assert "missing_tool" in missing_tool_result.content
    assert "list_tools" in missing_tool_result.content

    # Test with a tool registered directly in the registry (now always available)
    manager.runtime_registry.register_tool_obj(
        Tool(
            handler=simple_calculator,
            schema=calculator_schema,
            meta=get_tool_meta(simple_calculator),
        )
    )
    disabled_result = await manager.call_tool("disabled_tool", {"expression": "1+1"})
    assert not disabled_result.is_error
    assert disabled_result.content == "2"

    # Test error handling for invalid arguments - checking just that the error is set
    result = await manager.call_tool("calculator", {"wrong_arg": "value"})
    assert result.is_error
    # The specific error message could come directly from the tool if it returns a ToolResult
    # or be wrapped by our error handler, so we just check that it exists


@pytest.mark.asyncio
async def test_call_tool_creates_fd_when_enabled():
    """Test that call_tool wraps large results via FileDescriptorPlugin."""
    from unittest.mock import Mock

    from llmproc.plugins.file_descriptor import FileDescriptorPlugin

    manager = ToolManager()

    async def long_tool(**kwargs):
        return ToolResult.from_success("x" * 60)

    schema = {"name": "long_tool", "description": "Long output tool"}
    manager.runtime_registry.register_tool_obj(
        Tool(
            handler=long_tool,
            schema=schema,
            meta=get_tool_meta(long_tool),
        )
    )
    await manager.register_tools([long_tool], {})

    fd_manager = FileDescriptorManager(max_direct_output_chars=50)

    # Create FileDescriptorPlugin and enable it
    from llmproc.config.schema import FileDescriptorPluginConfig

    fd_plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
    fd_plugin.fd_manager = fd_manager

    # Create mock process with the FD plugin
    mock_process = Mock()
    mock_process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
    runner = PluginEventRunner(mock_process._submit_to_loop, [fd_plugin])
    mock_process.plugins = runner
    mock_process.hooks = runner

    runtime_context = {
        "process": mock_process,
        "fd_manager": fd_manager,
        "file_descriptor_enabled": True,
    }
    manager.set_runtime_context(runtime_context)

    result = await manager.call_tool("long_tool", {})

    assert isinstance(result, ToolResult)
    assert "<fd_result fd=" in result.content


@pytest.mark.asyncio
async def test_register_function_tool():
    """Test registering a function tool and verifying its behavior."""
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

    # Register the tool using function reference
    await manager.register_tools([double_value], {})

    # Initialize tools to register handlers and schemas
    config = {
        "fd_manager": None,
        "linked_programs": {},
        "linked_program_descriptions": {},
        "has_linked_programs": False,
        "provider": "test",
        "mcp_enabled": False,
    }
    result = await manager.register_tools([double_value], config)

    # Check the result is the manager itself (for chaining)
    assert result is manager

    # Verify the function tool is registered by using the getter method
    assert "double_value" in manager.registered_tools

    # Verify the tool is registered in the runtime registry
    assert "double_value" in manager.runtime_registry.get_tool_names()

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

    # Register both tools using function references
    await manager.register_tools([double_value, add_numbers], config)

    # Verify custom name is registered
    assert "custom_adder" in manager.registered_tools
    adder_handler = manager.runtime_registry.get_handler("custom_adder")
    add_result = await adder_handler(a=7, b=3)
    assert add_result.content == 10


@pytest.mark.asyncio
async def test_register_tools_directly():
    """Test registering tools directly with configuration."""
    manager = ToolManager()

    # Register the tools in the manager directly using function references
    from llmproc.config.schema import FileDescriptorPluginConfig

    plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
    await manager.register_tools(
        [
            calculator,
            read_file,
            fork_tool,
            spawn_tool,
            plugin.read_fd_tool,
            plugin.fd_to_file_tool,
        ],
        {},
    )

    # Create a mock process with properly mocked attributes
    mock_process = Mock()
    spawn_plugin = SpawnPlugin({"test_program": Mock()}, {"test_program": "Test program description"})
    runner = PluginEventRunner(lambda coro: asyncio.get_running_loop().create_task(coro), [spawn_plugin])
    mock_process.plugins = runner

    def get_plugin_side_effect(t):
        if t is FileDescriptorPlugin:
            return plugin
        if t is SpawnPlugin:
            return spawn_plugin
        return None

    mock_process.get_plugin.side_effect = get_plugin_side_effect
    mock_process._submit_to_loop = runner._submit
    mock_process.hooks = runner

    # Mock the fd_manager to avoid AttributeError
    mock_fd_manager = Mock()
    mock_process.get_plugin.return_value.fd_manager = mock_fd_manager

    # Set up the config dictionary with required fields
    config = {
        "fd_manager": mock_fd_manager,
        "linked_programs": mock_process.get_plugin(SpawnPlugin).linked_programs,
        "linked_program_descriptions": mock_process.get_plugin(SpawnPlugin).linked_program_descriptions,
        "has_linked_programs": True,
        "provider": "test",
    }

    # Register tools with configuration
    await manager.register_tools(
        [
            calculator,
            read_file,
            fork_tool,
            spawn_tool,
            plugin.read_fd_tool,
            plugin.fd_to_file_tool,
        ],
        config,
    )

    # Verify tools were registered by checking the runtime registry directly
    assert len(manager.runtime_registry.get_tool_names()) >= 6

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
        assert tool_name in manager.runtime_registry.get_tool_names()

    # Test calculator tool by calling it
    calculator_tool = manager.runtime_registry.get_tool("calculator")
    calculator_result = await calculator_tool.execute(
        {"expression": "2+2"},
        runtime_context=manager.runtime_context,
        process_access_level=manager.process_access_level,
    )
    assert isinstance(calculator_result, ToolResult)
    assert calculator_result.content == "4"
    assert not calculator_result.is_error

    # Test spawn tool (will return error in mock environment, but should be callable)
    # We need a proper mock context with a process that has linked_programs
    class Dummy:
        pass

    mock_program = MagicMock()
    mock_program.model_name = "dummy"

    mock_process = Dummy()
    mock_process.program = mock_program
    mock_process.plugins = [SpawnPlugin({"test_program": mock_program})]
    runtime_context = {"process": mock_process}
    # Runtime context must be included in the args dictionary
    args = {"program_name": "test_program", "prompt": "Test", "runtime_context": runtime_context}
    spawn_tool_obj = manager.runtime_registry.get_tool("spawn")
    spawn_result = await spawn_tool_obj.execute(
        {"program_name": "test_program", "prompt": "Test"},
        runtime_context=runtime_context,
        process_access_level=AccessLevel.ADMIN,
    )
    assert isinstance(spawn_result, ToolResult)
    assert spawn_result.is_error
    assert "provider" in str(spawn_result.content).lower()

    # Test that fork tool is registered but returns expected error for direct calls
    # Add runtime_context similarly to spawn tool
    fork_tool_obj = manager.runtime_registry.get_tool("fork")

    class DummyProc:
        pass

    mock_process = DummyProc()
    mock_process.iteration_state = None
    runtime_context = {"process": mock_process}

    fork_result = await fork_tool_obj.execute(
        {"prompts": ["Test"]},
        runtime_context=runtime_context,
        process_access_level=AccessLevel.ADMIN,
    )
    assert isinstance(fork_result, ToolResult)
    # Check that an appropriate error is surfaced (exact wording may evolve)
    assert fork_result.is_error


def test_register_tools_with_mixed_input():
    """Test registering tools from mixed config of strings and callables."""

    # Define a test function
    def test_func(x: int, y: int = 0) -> int:
        """Test function docstring."""
        return x + y

    # Create a tool manager
    manager = ToolManager()

    # Test with mixed input - using function references for builtin tools
    asyncio.run(manager.register_tools([calculator, test_func, read_file], {}))

    registered = manager.runtime_registry.get_tool_names()
    assert "calculator" in registered
    assert "read_file" in registered
    assert "test_func" in registered

    # Test with invalid item type - should raise ValueError
    with pytest.raises(ValueError):
        asyncio.run(manager.register_tools([calculator, 123, read_file], {}))

    # Test that valid tools still produce function_tools entries without error
    asyncio.run(manager.register_tools([calculator, read_file], {}))
    registered = manager.runtime_registry.get_tool_names()
    assert "calculator" in registered
    assert "read_file" in registered


def test_tool_registry_immutability():
    """Test that ToolRegistry methods return copies to prevent external modification."""
    # Create a registry
    registry = ToolRegistry()

    # Create a mock handler
    async def mock_handler(args):
        return ToolResult.from_success("Mock result")

    # Register a few tools
    registry.register_tool_obj(
        Tool(
            handler=mock_handler,
            schema={"name": "tool1", "description": "Tool 1"},
            meta=get_tool_meta(mock_handler),
        )
    )
    registry.register_tool_obj(
        Tool(
            handler=mock_handler,
            schema={"name": "tool2", "description": "Tool 2"},
            meta=get_tool_meta(mock_handler),
        )
    )

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
    assert "should_not_be_added" not in registry.get_tool_names()


@pytest.mark.asyncio
async def test_function_tool_returns_tool_result_without_double_wrap():
    """Ensure function tools can return ToolResult directly without wrapping."""
    manager = ToolManager()

    @register_tool(description="Echo result")
    async def echo_tool(x: int) -> ToolResult:
        return ToolResult.from_success(x)

    config = {
        "fd_manager": None,
        "linked_programs": {},
        "linked_program_descriptions": {},
        "has_linked_programs": False,
        "provider": "test",
        "mcp_enabled": False,
    }

    await manager.register_tools([echo_tool], config)

    result = await manager.call_tool("echo_tool", {"x": 5})

    assert isinstance(result, ToolResult)
    assert result.content == 5
    assert not isinstance(result.content, ToolResult)
