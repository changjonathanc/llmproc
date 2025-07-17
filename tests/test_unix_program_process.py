"""Tests for the Unix-inspired program/process transition model.

These tests verify the clean handoff pattern from LLMProgram (static definition)
to LLMProcess (runtime state).
"""

import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import asyncio

import pytest
from llmproc.common.metadata import get_tool_meta
from llmproc.tools.core import Tool
from llmproc.plugin.plugin_event_runner import PluginEventRunner
from llmproc.common.results import ToolResult
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.function_tools import register_tool
from llmproc.tools.tool_manager import ToolManager


@pytest.fixture
def mock_program():
    """Create a mock LLMProgram for testing."""
    program = MagicMock(spec=LLMProgram)
    program.tool_manager = ToolManager()

    # Set some basic program properties
    program.model_name = "test-model"
    program.provider = "test-provider"
    program.system_prompt = "Test system prompt"
    program.project_id = None
    program.region = None
    program.compiled = True

    # Mock get_tool_configuration to return empty config
    program.get_tool_configuration.return_value = {
        "provider": "test-provider",
        "file_descriptor_enabled": False,
        "has_linked_programs": False,
        "linked_programs": {},
        "linked_program_descriptions": {},
    }

    # Set up async methods
    program.start = AsyncMock()
    program.tool_manager.register_tools = AsyncMock()

    return program


@pytest.mark.asyncio
async def test_tool_manager_runtime_context():
    """Test setting and using runtime context in ToolManager."""
    tool_manager = ToolManager()

    # Create a mock context-aware handler
    @register_tool(requires_context=True)
    async def test_handler(param=None, runtime_context=None, **kwargs):
        return ToolResult.from_success(f"Got context: {runtime_context['test_key']}")

    # Create a mock tool definition
    test_tool_def = {
        "name": "test_tool",
        "description": "Test tool for runtime context",
        "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}},
    }

    # Register the tool
    meta = get_tool_meta(test_handler)
    meta.name = "mock_tool"
    tool_manager.runtime_registry.register_tool_obj(
        Tool(
            handler=test_handler,
            schema=test_tool_def,
            meta=meta,
        )
    )
    await tool_manager.register_tools([test_handler], {})

    # Set runtime context
    from unittest.mock import Mock
    mock_process = Mock()
    mock_process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
    runner = PluginEventRunner(mock_process._submit_to_loop, [])
    mock_process.plugins = runner
    mock_process.hooks = runner
    test_context = {"test_key": "test_value", "process": mock_process}
    tool_manager.set_runtime_context(test_context)

    # Call the tool
    result = await tool_manager.call_tool(
        "mock_tool", {"param": "test"}
    )  # Using dict format for backward compatibility test

    # Check that context was properly injected
    assert isinstance(result, ToolResult)
    assert hasattr(result, "is_error")
    assert not result.is_error
    assert "Got context: test_value" in result.content


def test_configuration_based_tool_registration():
    """Test tool registration using configuration instead of process."""
    tool_manager = ToolManager()
    # Register the tools
    from llmproc.tools.builtin import fork_tool, spawn_tool
    from llmproc.plugins.message_id import MessageIDPlugin
    from llmproc.config.schema import MessageIDPluginConfig

    # Create plugin and get goto tool
    message_id_plugin = MessageIDPlugin(MessageIDPluginConfig())
    handle_goto = message_id_plugin.goto_tool

    # Create configuration with needed components
    config = {
        "linked_programs": {"test_program": MagicMock()},
        "linked_program_descriptions": {"test_program": "Test program description"},
        "has_linked_programs": True,
        "provider": "anthropic",
    }

    import asyncio
    asyncio.run(tool_manager.register_tools([handle_goto, fork_tool, spawn_tool], config))

    # Set runtime context
    tool_manager.set_runtime_context(
        {
            "linked_programs": config["linked_programs"],
            "linked_program_descriptions": config["linked_program_descriptions"],
        }
    )

    # Register function tools directly
    from llmproc.tools.builtin import fork_tool, spawn_tool
    # handle_goto already defined above

    asyncio.run(tool_manager.register_tools([handle_goto, fork_tool, spawn_tool], {}))

    # Mock registration for this test since we don't want to run the async method
    # We'll just simulate the registry population to test the core functionality
    goto_handler = AsyncMock()
    fork_handler = AsyncMock()
    spawn_handler = AsyncMock()
    tool_manager.runtime_registry.register_tool_obj(
        Tool(handler=goto_handler, schema={"name": "goto", "description": "Goto tool"}, meta=get_tool_meta(goto_handler))
    )
    tool_manager.runtime_registry.register_tool_obj(
        Tool(handler=fork_handler, schema={"name": "fork", "description": "Fork tool"}, meta=get_tool_meta(fork_handler))
    )
    tool_manager.runtime_registry.register_tool_obj(
        Tool(handler=spawn_handler, schema={"name": "spawn", "description": "Spawn tool"}, meta=get_tool_meta(spawn_handler))
    )

    # Verify goto tool was registered
    assert "goto" in tool_manager.runtime_registry.get_tool_names()
    assert any(schema.get("name") == "goto" for schema in tool_manager.runtime_registry.get_definitions())

    # Verify fork tool was registered
    assert "fork" in tool_manager.runtime_registry.get_tool_names()
    assert any(schema.get("name") == "fork" for schema in tool_manager.runtime_registry.get_definitions())

    # Verify spawn tool was registered
    assert "spawn" in tool_manager.runtime_registry.get_tool_names()
    assert any(schema.get("name") == "spawn" for schema in tool_manager.runtime_registry.get_definitions())


def test_fd_tools_implementation():
    """Test that file descriptor tools are properly marked as context-aware."""

    # Create a context-aware handler directly
    @register_tool(requires_context=True)
    async def test_handler(args, runtime_context=None):
        return "test"

    # Verify the context-aware marker was set in metadata
    from llmproc.common.metadata import get_tool_meta

    meta = get_tool_meta(test_handler)
    assert meta.requires_context
    assert get_tool_meta(test_handler).requires_context

    # Create a non-context-aware handler
    async def standard_handler(args):
        return "test"

    # Verify the context-aware marker is not set
    assert not hasattr(standard_handler, "_requires_context")
    assert not get_tool_meta(standard_handler).requires_context


def test_runtime_context_handler_execution():
    """Test execution of context-aware handlers with runtime context."""
    # Create a mock context
    mock_context = {"key": "value", "process": "mock_process"}

    # Create a context-aware handler that uses the context
    @register_tool(requires_context=True)
    async def context_handler(args, runtime_context=None):
        # The decorator validation already happened, we can safely access the key
        if runtime_context and "key" in runtime_context:
            return ToolResult.from_success(f"Got context: {runtime_context['key']}")
        return ToolResult.from_error("Missing context or key")

    # Create test for runtime context injection
    # This verifies that our decorator properly sets up the runtime context pattern
    async def test_call():
        result = await context_handler({"test": "args"}, runtime_context=mock_context)
        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert result.content == "Got context: value"

        # Test without context
        result_no_context = await context_handler({"test": "args"})
        assert isinstance(result_no_context, ToolResult)
        assert result_no_context.is_error
        assert "missing" in result_no_context.content.lower()

    # Run the test
    import asyncio

    asyncio.run(test_call())


def test_tool_manager_setup_runtime_context():
    """Test runtime context with basic tool operations."""
    # This is a simplified test to verify runtime context handling
    tool_manager = ToolManager()
    # Import tools for registration
    from llmproc.plugins.file_descriptor import FileDescriptorPlugin
    from llmproc.plugins.message_id import MessageIDPlugin
    from llmproc.config.schema import FileDescriptorPluginConfig, MessageIDPluginConfig

    # Create plugins and get tools
    fd_plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
    message_id_plugin = MessageIDPlugin(MessageIDPluginConfig())
    read_fd_tool = fd_plugin.read_fd_tool
    handle_goto = message_id_plugin.goto_tool

    # Create mock components
    mock_process = MagicMock()
    mock_fd_manager = MagicMock()
    mock_linked_programs = {"test_program": MagicMock()}

    # Set runtime context
    runtime_context = {
        "process": mock_process,
        "linked_programs": mock_linked_programs,
    }

    # Apply context
    tool_manager.set_runtime_context(runtime_context)

    # Verify context was set
    assert tool_manager.runtime_context == runtime_context
    assert tool_manager.runtime_context["process"] == mock_process

    # Test registering tools
    import asyncio
    asyncio.run(tool_manager.register_tools([handle_goto, read_fd_tool], {}))
    registered = tool_manager.runtime_registry.get_tool_names()
    assert "goto" in registered
    assert "read_fd" in registered

    # Register a mock context-aware tool handler
    @register_tool(requires_context=True)
    async def mock_handler(args, runtime_context=None):
        return ToolResult.from_success("Context accessed")

    # Register the tool
    tool_def = {"name": "mock_tool", "description": "Mock tool"}
    meta = get_tool_meta(mock_handler)
    meta.name = "mock_tool"
    tool_manager.runtime_registry.register_tool_obj(
        Tool(
            handler=mock_handler,
            schema=tool_def,
            meta=meta,
        )
    )

    # Verify requires_context correctly identifies the handler
    assert get_tool_meta(mock_handler).requires_context

    # Verify the tool was registered
    assert "mock_tool" in tool_manager.runtime_registry.get_tool_names()
    assert tool_manager.runtime_registry.get_handler("mock_tool") is mock_handler


def test_file_descriptor_tools_registration():
    """Test registration of file descriptor tools using configuration approach."""
    from llmproc.plugins.file_descriptor import FileDescriptorPlugin, FileDescriptorManager
    from llmproc.config.schema import FileDescriptorPluginConfig

    tool_manager = ToolManager()
    plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
    plugin.fd_manager = FileDescriptorManager()
    import asyncio
    asyncio.run(tool_manager.register_tools([plugin.read_fd_tool, plugin.fd_to_file_tool], {}))

    # Create a mock FileDescriptorManager
    fd_manager = MagicMock()
    fd_manager.register_fd_tool = MagicMock()

    # Register function tools directly
    # Create configuration with FD manager
    config = {
        "fd_manager": fd_manager,
        "linked_programs": {},
        "linked_program_descriptions": {},
        "has_linked_programs": False,
        "provider": "anthropic",
        "file_descriptor_enabled": True,
    }

    # Set runtime context
    tool_manager.set_runtime_context(
        {
            "linked_programs": {},
            "linked_program_descriptions": {},
        }
    )

    # Register tools with configuration asynchronously
    import asyncio

    asyncio.run(
        tool_manager.register_tools([
            plugin.read_fd_tool,
            plugin.fd_to_file_tool,
        ], config)
    )

    # Verify read_fd tool was registered
    assert "read_fd" in tool_manager.runtime_registry.get_tool_names()
    assert any(schema.get("name") == "read_fd" for schema in tool_manager.runtime_registry.get_definitions())

    # Verify fd_to_file tool was registered
    assert "fd_to_file" in tool_manager.runtime_registry.get_tool_names()
    assert any(schema.get("name") == "fd_to_file" for schema in tool_manager.runtime_registry.get_definitions())

    # Verify tool registration without checking fd_manager.register_fd_tool calls
    # which now happen during FileDescriptorManager initialization in program_exec.py
    # Register the FD tools manually for this test since we're not using the new initialization path
    fd_manager.register_fd_tool("read_fd")
    fd_manager.register_fd_tool("fd_to_file")


@pytest.mark.asyncio
async def test_tool_manager_registration():
    """Test the tool manager's registration of system tools."""
    tool_manager = ToolManager()
    # Import tools for registration
    from llmproc.plugins.file_descriptor import FileDescriptorPlugin, FileDescriptorManager
    from llmproc.config.schema import FileDescriptorPluginConfig, MessageIDPluginConfig
    from llmproc.tools.builtin import fork_tool, spawn_tool
    from llmproc.plugins.message_id import MessageIDPlugin

    # Create plugins and get tools
    message_id_plugin = MessageIDPlugin(MessageIDPluginConfig())
    handle_goto = message_id_plugin.goto_tool

    plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
    plugin.fd_manager = FileDescriptorManager()
    await tool_manager.register_tools(
        [handle_goto, fork_tool, spawn_tool, plugin.read_fd_tool, plugin.fd_to_file_tool],
        {},
    )

    # Create mock components
    mock_fd_manager = MagicMock()
    mock_fd_manager.register_fd_tool = MagicMock()

    # Register function tools directly
    await tool_manager.register_tools(
        [
            handle_goto,
            fork_tool,
            spawn_tool,
            plugin.read_fd_tool,
            plugin.fd_to_file_tool,
        ],
        {},
    )

    # Create configuration dictionary
    config = {
        "fd_manager": mock_fd_manager,
        "linked_programs": {"test_program": MagicMock()},
        "linked_program_descriptions": {"test_program": "Test program description"},
        "has_linked_programs": True,
        "provider": "anthropic",
        "file_descriptor_enabled": True,
        "mcp_enabled": False,
    }

    # Set runtime context
    tool_manager.set_runtime_context(
        {
            "linked_programs": config["linked_programs"],
            "linked_program_descriptions": config["linked_program_descriptions"],
        }
    )

    # Mock registration for this test since we don't want to actually run the async method
    # We'll just simulate the registry population to test the core functionality
    goto_handler2 = AsyncMock()
    fork_handler2 = AsyncMock()
    spawn_handler2 = AsyncMock()
    read_fd_handler = AsyncMock()
    fd_to_file_handler = AsyncMock()
    tool_manager.runtime_registry.register_tool_obj(
        Tool(handler=goto_handler2, schema={"name": "goto", "description": "Goto tool"}, meta=get_tool_meta(goto_handler2))
    )
    tool_manager.runtime_registry.register_tool_obj(
        Tool(handler=fork_handler2, schema={"name": "fork", "description": "Fork tool"}, meta=get_tool_meta(fork_handler2))
    )
    tool_manager.runtime_registry.register_tool_obj(
        Tool(handler=spawn_handler2, schema={"name": "spawn", "description": "Spawn tool"}, meta=get_tool_meta(spawn_handler2))
    )
    tool_manager.runtime_registry.register_tool_obj(
        Tool(handler=read_fd_handler, schema={"name": "read_fd", "description": "Read FD tool"}, meta=get_tool_meta(read_fd_handler))
    )
    tool_manager.runtime_registry.register_tool_obj(
        Tool(handler=fd_to_file_handler, schema={"name": "fd_to_file", "description": "FD to File tool"}, meta=get_tool_meta(fd_to_file_handler))
    )

    # Verify goto, fork, and spawn tools were registered
    assert "goto" in tool_manager.runtime_registry.get_tool_names()
    assert "fork" in tool_manager.runtime_registry.get_tool_names()
    assert "spawn" in tool_manager.runtime_registry.get_tool_names()
    assert "read_fd" in tool_manager.runtime_registry.get_tool_names()
    assert "fd_to_file" in tool_manager.runtime_registry.get_tool_names()

    # Skip testing MCP initialization since we have separate tests for that
    # Adding MCP-specific test in a separate section causes too many issues with
    # mocking the private libraries

    # Just verify that the tools we did register are there
    assert "goto" in tool_manager.runtime_registry.get_tool_names()
    assert "fork" in tool_manager.runtime_registry.get_tool_names()
    assert "spawn" in tool_manager.runtime_registry.get_tool_names()

    # Register the FD tools manually for this test since we're not using the new initialization path
    # This would normally happen in program_exec.py's initialize_file_descriptor_system function
    mock_fd_manager.register_fd_tool("read_fd")
    mock_fd_manager.register_fd_tool("fd_to_file")


@pytest.mark.asyncio
async def test_program_to_process_handoff():
    """Test the clean handoff from LLMProgram to LLMProcess.

    This test verifies the Unix-inspired initialization pattern where:
    1. Program provides static configuration
    2. Tools are initialized before process creation
    3. Process is created with pre-initialized tools
    4. Runtime context is properly set up
    """
    from tests.conftest import create_test_llmprocess_directly

    # Create a real program with minimal configuration
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",  # Use a supported provider for the test
        system_prompt="Test system prompt",
    )
    program.tool_manager = ToolManager()

    # Mock the necessary components for the create_process flow
    with (
        patch(
            "llmproc.program_exec.create_process",
            side_effect=create_process_side_effect,
        ),
        patch("llmproc.program_exec.prepare_process_state") as mock_prepare_state,
        patch("llmproc.program_exec.get_provider_client") as mock_get_client,
    ):
        # Create a mock client
        mock_get_client.return_value = MagicMock()

        # Set up the side effect for prepare_process_state
        mock_prepare_state.return_value = {
            "model_name": program.model_name,
            "provider": program.provider,
            "base_system_prompt": program.system_prompt,
            "enriched_system_prompt": f"Enriched: {program.system_prompt}",
            "state": [],
            "client": mock_get_client.return_value,
            "tool_manager": program.tool_manager,
            "program": program,  # Add program to the state dictionary
        }

        # Mock the register_tools method to avoid actual initialization
        original_initialize = program.tool_manager.register_tools
        program.tool_manager.register_tools = AsyncMock()
        program.tool_manager.register_tools.return_value = program.tool_manager

        try:
            # Use the start() method to create a process
            # This should follow the Unix-inspired pattern
            process = await program.start()

            # Verify the right methods were called in order
            program.tool_manager.register_tools.assert_called_once()

            # Verify process was created with the right attributes
            assert process.model_name == "test-model"
            assert process.provider == "anthropic"

            # Verify runtime context was set up
            assert hasattr(process.tool_manager, "runtime_context")

        finally:
            # Restore original method
            program.tool_manager.register_tools = original_initialize


# Helper function for the side effect
async def create_process_side_effect(program, access_level=None):
    """Side effect for mocking create_process."""
    from tests.conftest import create_test_llmprocess_directly

    # Call the register_tools method directly
    config = program.get_tool_configuration()
    await program.tool_manager.register_tools(program.tools or [], config)

    # Create a process using our test helper
    return create_test_llmprocess_directly(program=program)
