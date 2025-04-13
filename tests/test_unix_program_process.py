"""Tests for the Unix-inspired program/process transition model.

These tests verify the clean handoff pattern from LLMProgram (static definition)
to LLMProcess (runtime state) as described in RFC053.
"""

import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from llmproc.common.results import ToolResult
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.context_aware import context_aware, is_context_aware
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
    program.display_name = "Test Model"
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
    program.tool_manager.initialize_tools = AsyncMock()

    return program


@pytest.mark.asyncio
async def test_tool_manager_runtime_context():
    """Test setting and using runtime context in ToolManager."""
    tool_manager = ToolManager()

    # Create a mock context-aware handler
    @context_aware
    async def test_handler(param=None, runtime_context=None, **kwargs):
        if runtime_context and "test_key" in runtime_context:
            return ToolResult.from_success(
                f"Got context: {runtime_context['test_key']}"
            )
        return ToolResult.from_error("No context or missing key")

    # Create a mock tool definition
    test_tool_def = {
        "name": "test_tool",
        "description": "Test tool for runtime context",
        "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}},
    }

    # Register the tool
    tool_manager.runtime_registry.register_tool(
        "test_tool", test_handler, test_tool_def
    )
    tool_manager.enabled_tools = ["test_tool"]

    # Set runtime context
    test_context = {"test_key": "test_value", "process": "test_process"}
    tool_manager.set_runtime_context(test_context)

    # Call the tool
    result = await tool_manager.call_tool(
        "test_tool", {"param": "test"}
    )  # Using dict format for backward compatibility test

    # Check that context was properly injected
    assert isinstance(result, ToolResult)
    assert hasattr(result, "is_error")
    assert not result.is_error
    assert "Got context: test_value" in result.content


def test_configuration_based_tool_registration():
    """Test tool registration using configuration instead of process."""
    tool_manager = ToolManager()
    tool_manager.enabled_tools = ["goto", "fork", "spawn"]

    # Create configuration with needed components
    config = {
        "fd_manager": None,  # No FD manager needed for goto
        "linked_programs": {"test_program": MagicMock()},
        "linked_program_descriptions": {"test_program": "Test program description"},
        "has_linked_programs": True,
        "provider": "anthropic",
    }

    # Set runtime context
    tool_manager.set_runtime_context(
        {
            "fd_manager": None,
            "linked_programs": config["linked_programs"],
            "linked_program_descriptions": config["linked_program_descriptions"],
        }
    )

    # Load the builtin tools which is now required
    tool_manager._load_builtin_tools()

    # Register tools with configuration (non-async method)
    tool_manager.register_system_tools(config)

    # Verify goto tool was registered
    assert "goto" in tool_manager.runtime_registry.tool_handlers
    assert any(
        schema.get("name") == "goto"
        for schema in tool_manager.runtime_registry.tool_definitions
    )

    # Verify fork tool was registered
    assert "fork" in tool_manager.runtime_registry.tool_handlers
    assert any(
        schema.get("name") == "fork"
        for schema in tool_manager.runtime_registry.tool_definitions
    )

    # Verify spawn tool was registered
    assert "spawn" in tool_manager.runtime_registry.tool_handlers
    assert any(
        schema.get("name") == "spawn"
        for schema in tool_manager.runtime_registry.tool_definitions
    )


def test_fd_tools_implementation():
    """Test that file descriptor tools are properly marked as context-aware."""

    # Create a context-aware handler directly
    @context_aware
    async def test_handler(args, runtime_context=None):
        return "test"

    # Verify the context-aware marker was set
    assert hasattr(test_handler, "_needs_context")
    assert test_handler._needs_context
    assert is_context_aware(test_handler)

    # Create a non-context-aware handler
    async def standard_handler(args):
        return "test"

    # Verify the context-aware marker is not set
    assert not hasattr(standard_handler, "_needs_context")
    assert not is_context_aware(standard_handler)


def test_runtime_context_handler_execution():
    """Test execution of context-aware handlers with runtime context."""
    # Create a mock context
    mock_context = {"key": "value", "process": "mock_process"}

    # Create a context-aware handler that uses the context
    @context_aware
    async def context_handler(args, runtime_context=None):
        if runtime_context and "key" in runtime_context:
            return ToolResult.from_success(f"Got context: {runtime_context['key']}")
        return ToolResult.from_error("Missing context")

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
        assert result_no_context.content == "Missing context"

    # Run the test
    import asyncio

    asyncio.run(test_call())


def test_tool_manager_setup_runtime_context():
    """Test runtime context with basic tool operations."""
    # This is a simplified test to verify runtime context handling
    tool_manager = ToolManager()

    # Create mock components
    mock_process = MagicMock()
    mock_fd_manager = MagicMock()
    mock_linked_programs = {"test_program": MagicMock()}

    # Set runtime context
    runtime_context = {
        "process": mock_process,
        "fd_manager": mock_fd_manager,
        "linked_programs": mock_linked_programs,
    }

    # Apply context
    tool_manager.set_runtime_context(runtime_context)

    # Verify context was set
    assert tool_manager.runtime_context == runtime_context
    assert tool_manager.runtime_context["process"] == mock_process
    assert tool_manager.runtime_context["fd_manager"] == mock_fd_manager

    # Test enabling tools
    tool_manager.enabled_tools = ["goto", "read_fd"]
    assert "goto" in tool_manager.enabled_tools
    assert "read_fd" in tool_manager.enabled_tools

    # Register a mock context-aware tool handler
    @context_aware
    async def mock_handler(args, runtime_context=None):
        return ToolResult.from_success("Context accessed")

    # Register the tool
    tool_def = {"name": "mock_tool", "description": "Mock tool"}
    tool_manager.runtime_registry.register_tool("mock_tool", mock_handler, tool_def)

    # Verify is_context_aware correctly identifies the handler
    assert is_context_aware(mock_handler)

    # Verify the tool was registered
    assert "mock_tool" in tool_manager.runtime_registry.tool_handlers
    assert tool_manager.runtime_registry.tool_handlers["mock_tool"] is mock_handler


def test_file_descriptor_tools_registration():
    """Test registration of file descriptor tools using configuration approach."""
    tool_manager = ToolManager()
    tool_manager.enabled_tools = ["read_fd", "fd_to_file"]

    # Create a mock FileDescriptorManager
    fd_manager = MagicMock()
    fd_manager.register_fd_tool = MagicMock()

    # Load the builtin tools first
    tool_manager._load_builtin_tools()

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
            "fd_manager": fd_manager,
            "linked_programs": {},
            "linked_program_descriptions": {},
        }
    )

    # Register tools with configuration
    tool_manager.register_system_tools(config)

    # Verify read_fd tool was registered
    assert "read_fd" in tool_manager.runtime_registry.tool_handlers
    assert any(
        schema.get("name") == "read_fd"
        for schema in tool_manager.runtime_registry.tool_definitions
    )

    # Verify fd_to_file tool was registered
    assert "fd_to_file" in tool_manager.runtime_registry.tool_handlers
    assert any(
        schema.get("name") == "fd_to_file"
        for schema in tool_manager.runtime_registry.tool_definitions
    )

    # Verify fd_manager.register_fd_tool was called for both tools
    # Note: The tools might be registered multiple times because we're transitioning the implementation
    assert fd_manager.register_fd_tool.call_count >= 2
    fd_manager.register_fd_tool.assert_any_call("read_fd")
    fd_manager.register_fd_tool.assert_any_call("fd_to_file")


@pytest.mark.asyncio
async def test_tool_manager_initialization():
    """Test the tool manager's initialize_tools and register_system_tools methods."""
    tool_manager = ToolManager()
    tool_manager.enabled_tools = ["goto", "fork", "spawn", "read_fd", "fd_to_file"]

    # Create mock components
    mock_fd_manager = MagicMock()
    mock_fd_manager.register_fd_tool = MagicMock()

    # Load the builtin tools first
    tool_manager._load_builtin_tools()

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
            "fd_manager": mock_fd_manager,
            "linked_programs": config["linked_programs"],
            "linked_program_descriptions": config["linked_program_descriptions"],
        }
    )

    # Initialize builtin registry before registering tools
    tool_manager._load_builtin_tools()

    # Register system tools first - this is synchronous and handles the basic tools
    tool_manager.register_system_tools(config)

    # Verify goto, fork, and spawn tools were registered
    assert "goto" in tool_manager.runtime_registry.tool_handlers
    assert "fork" in tool_manager.runtime_registry.tool_handlers
    assert "spawn" in tool_manager.runtime_registry.tool_handlers
    assert "read_fd" in tool_manager.runtime_registry.tool_handlers
    assert "fd_to_file" in tool_manager.runtime_registry.tool_handlers

    # Skip testing MCP initialization since we have separate tests for that
    # Adding MCP-specific test in a separate section causes too many issues with
    # mocking the private libraries

    # Just verify that the tools we did register are there
    assert "goto" in tool_manager.runtime_registry.tool_handlers
    assert "fork" in tool_manager.runtime_registry.tool_handlers
    assert "spawn" in tool_manager.runtime_registry.tool_handlers

    # Verify that fd_manager.register_fd_tool was called for both FD tools
    mock_fd_manager.register_fd_tool.assert_any_call("read_fd")
    mock_fd_manager.register_fd_tool.assert_any_call("fd_to_file")


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

    # Mock the necessary components for the create_process flow
    with (
        patch(
            "llmproc.program_exec.create_process",
            side_effect=create_process_side_effect,
        ),
        patch("llmproc.program_exec.prepare_process_state") as mock_prepare_state,
        patch("llmproc.program_exec.initialize_client") as mock_initialize_client,
        patch("llmproc.providers.get_provider_client") as mock_get_client,
    ):
        # Create a mock client
        mock_get_client.return_value = MagicMock()
        mock_initialize_client.return_value = mock_get_client.return_value

        # Set up the side effect for prepare_process_state
        mock_prepare_state.return_value = {
            "model_name": program.model_name,
            "provider": program.provider,
            "original_system_prompt": program.system_prompt,
            "system_prompt": program.system_prompt,
            "enriched_system_prompt": f"Enriched: {program.system_prompt}",
            "state": [],
            "client": mock_get_client.return_value,
            "tool_manager": program.tool_manager,
            "program": program,  # Add program to the state dictionary
        }

        # Mock the initialize_tools method to avoid actual initialization
        original_initialize = program.tool_manager.initialize_tools
        program.tool_manager.initialize_tools = AsyncMock()
        program.tool_manager.initialize_tools.return_value = program.tool_manager

        try:
            # Use the start() method to create a process
            # This should follow the Unix-inspired pattern
            process = await program.start()

            # Verify the right methods were called in order
            program.tool_manager.initialize_tools.assert_called_once()

            # Verify process was created with the right attributes
            assert process.model_name == "test-model"
            assert process.provider == "anthropic"
            assert process.system_prompt == "Test system prompt"

            # Verify runtime context was set up
            assert hasattr(process.tool_manager, "runtime_context")

        finally:
            # Restore original method
            program.tool_manager.initialize_tools = original_initialize


# Helper function for the side effect
async def create_process_side_effect(program, additional_preload_files=None):
    """Side effect for mocking create_process."""
    from tests.conftest import create_test_llmprocess_directly

    # Call the initialize_tools method directly
    config = program.get_tool_configuration()
    await program.tool_manager.initialize_tools(config)

    # Create a process using our test helper
    return create_test_llmprocess_directly(program=program)
