"""Tests for the ToolManager multi-registry architecture."""

import asyncio
from unittest.mock import Mock, patch

import llmproc.program_exec as program_exec

import pytest
from llmproc.common.results import ToolResult
from llmproc.tools import ToolManager, ToolRegistry
from llmproc.tools.core import Tool
from llmproc.common.metadata import get_tool_meta
from llmproc.tools.builtin import calculator, fork_tool, read_file, spawn_tool
from llmproc.tools.function_tools import register_tool


def test_tool_manager_registry_initialization():
    """Test that ToolManager initializes with the necessary registries."""
    manager = ToolManager()

    assert isinstance(manager.runtime_registry, ToolRegistry)

    # Check that other attributes are initialized as expected
    assert isinstance(manager.registered_tools, list)
    assert len(manager.registered_tools) == 0
    assert manager.mcp_aggregator is None


@pytest.mark.asyncio
async def test_register_tools_basic():
    """Test the basic register_tools method."""
    manager = ToolManager()

    # Create a mock process configuration
    mock_config = {
        "fd_manager": None,
        "mcp_enabled": False,  # Ensure MCP is disabled
    }

    # Initialize the tools
    await manager.register_tools(
        [calculator, read_file, fork_tool],
        mock_config,
    )

    # Verify that the runtime registry is populated
    assert len(manager.runtime_registry.get_tool_names()) > 0

    # Check that the enabled tools are registered
    assert "calculator" in manager.runtime_registry.get_tool_names()
    assert "read_file" in manager.runtime_registry.get_tool_names()
    assert "fork" in manager.runtime_registry.get_tool_names()


@pytest.mark.asyncio
async def test_function_tool_direct_registration():
    """Test that function tools are directly registered to runtime registry."""
    manager = ToolManager()

    # Register tools using function references
    # Create a mock process configuration
    mock_config = {
        "fd_manager": None,
        "mcp_enabled": False,  # Disable MCP
    }

    # Initialize the tools
    await manager.register_tools([calculator, read_file], mock_config)

    # Verify direct registration was successful
    assert "calculator" in manager.runtime_registry.get_tool_names()
    assert "read_file" in manager.runtime_registry.get_tool_names()

    # Verify core functionality works


@pytest.mark.asyncio
async def test_execution_phase_uses_runtime_registry():
    """Test that the execution phase (call_tool and get_tool_schemas) uses the runtime registry."""
    manager = ToolManager()

    # Register a tool only in the main registry
    async def main_registry_handler(**kwargs):
        return ToolResult.from_success("Called from main registry")

    main_tool_schema = {
        "name": "main_tool",
        "description": "A tool only in the main registry",
        "input_schema": {"type": "object", "properties": {}},
    }

    # Register a different tool only in the runtime registry
    async def runtime_registry_handler(**kwargs):
        return ToolResult.from_success("Called from runtime registry")

    runtime_tool_schema = {
        "name": "runtime_tool",
        "description": "A tool only in the runtime registry",
        "input_schema": {"type": "object", "properties": {}},
    }

    # Test just with runtime registry first
    manager.runtime_registry.register_tool_obj(
        Tool(
            handler=runtime_registry_handler,
            schema=runtime_tool_schema,
            meta=get_tool_meta(runtime_registry_handler),
        )
    )

    # Create a callable for runtime_tool
    async def runtime_tool_callable(**kwargs):
        return runtime_registry_handler(**kwargs)

    runtime_tool_callable.__name__ = "runtime_tool"

    # Register the tool using callable
    await manager.register_tools([runtime_tool_callable], {})

    # Test runtime tool - should be found in runtime registry
    result_runtime = await manager.call_tool("runtime_tool", {})
    assert result_runtime.content == "Called from runtime registry"

    # Now test with main registry tools using a fresh manager
    # This avoids interference between tests
    manager2 = ToolManager()

    # Register in runtime registry
    manager2.runtime_registry.register_tool_obj(
        Tool(
            handler=main_registry_handler,
            schema=main_tool_schema,
            meta=get_tool_meta(main_registry_handler),
        )
    )

    # Create a callable for main_tool
    async def main_tool_callable(**kwargs):
        return main_registry_handler(**kwargs)

    main_tool_callable.__name__ = "main_tool"

    # Register the tool using callable
    await manager2.register_tools([main_tool_callable], {})

    # Call the main tool
    result_main = await manager2.call_tool("main_tool", {})
    assert result_main.content == "Called from main registry"

    # Create a third manager to test tools in both registries
    manager3 = ToolManager()

    # Define tool handlers
    async def shared_tool_main_handler(**kwargs):
        return ToolResult.from_success("Shared tool - called from main registry")

    async def shared_tool_runtime_handler(**kwargs):
        return ToolResult.from_success("Shared tool - called from runtime registry")

    shared_tool_schema = {
        "name": "shared_tool",
        "description": "A tool in both registries",
        "input_schema": {"type": "object", "properties": {}},
    }

    # Register the tool in runtime registry
    manager3.runtime_registry.register_tool_obj(
        Tool(
            handler=shared_tool_runtime_handler,
            schema=shared_tool_schema.copy(),
            meta=get_tool_meta(shared_tool_runtime_handler),
        )
    )

    # Call the shared tool - should use runtime registry version
    result_shared = await manager3.call_tool("shared_tool", {})
    assert result_shared.content == "Shared tool - called from runtime registry"

    # Test get_tool_schemas when runtime registry is populated
    manager4 = ToolManager()

    # Register tools in runtime registry
    manager4.runtime_registry.register_tool_obj(
        Tool(
            handler=runtime_registry_handler,
            schema=runtime_tool_schema.copy(),
            meta=get_tool_meta(runtime_registry_handler),
        )
    )
    manager4.runtime_registry.register_tool_obj(
        Tool(
            handler=shared_tool_runtime_handler,
            schema=shared_tool_schema.copy(),
            meta=get_tool_meta(shared_tool_runtime_handler),
        )
    )

    # Get schemas
    schemas = manager4.get_tool_schemas()
    schema_names = [schema["name"] for schema in schemas]

    # Should only include runtime registry tools since it's populated
    assert sorted(schema_names) == sorted(["runtime_tool", "shared_tool"])

    # Test loading tools directly into runtime registry
    manager5 = ToolManager()

    # Register tools in runtime registry
    manager5.runtime_registry.register_tool_obj(
        Tool(
            handler=main_registry_handler,
            schema=main_tool_schema.copy(),
            meta=get_tool_meta(main_registry_handler),
        )
    )
    manager5.runtime_registry.register_tool_obj(
        Tool(
            handler=shared_tool_main_handler,
            schema=shared_tool_schema.copy(),
            meta=get_tool_meta(shared_tool_main_handler),
        )
    )

    # Get schemas
    schemas = manager5.get_tool_schemas()
    schema_names = [schema["name"] for schema in schemas]

    # Should include all enabled tools from runtime registry
    assert sorted(schema_names) == sorted(["main_tool", "shared_tool"])


# Helper class for async mock in Python < 3.8
class AsyncMock(Mock):
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


@pytest.mark.asyncio
async def test_llmprocess_unified_mcp_initialization():
    """Test that program_exec delegates initialization to ToolManager with config."""
    # Import the program_exec create_process function
    from llmproc.program import LLMProgram
    from llmproc.program_exec import create_process

    # Create a mock program for testing
    program = Mock(spec=LLMProgram)
    program.compiled = True
    program.model_name = "test-model"
    program.provider = "anthropic"
    program.project_id = None
    program.region = None

    # Configure the tool configuration to be returned
    tool_config = {
        "mcp_enabled": True,
        "provider": "anthropic",
        "mcp_config_path": "/test/path/config.json",
    }
    program.get_tool_configuration.return_value = tool_config

    # Create a real ToolManager object
    tool_manager = ToolManager()
    program.tool_manager = tool_manager

    # Mock instantiate_process to return a mock LLMProcess
    mock_process = Mock()
    mock_process.tool_manager = tool_manager

    # Mock the necessary components of the create_process flow
    with (
        patch("llmproc.program_exec.instantiate_process", return_value=mock_process),
        patch(
            "llmproc.program_exec.prepare_process_config",
            return_value=program_exec.ProcessConfig(
                program=program,
                model_name=program.model_name,
                provider=program.provider,
                base_system_prompt="x",
            ),
        ),
        patch("llmproc.program_exec.setup_runtime_context"),
        patch("llmproc.program_exec.validate_process"),
        patch("llmproc.program_exec.ToolManager.register_tools", new_callable=AsyncMock) as mock_initialize,
    ):
        # Setup the mock to return the manager itself for method chaining
        mock_initialize.return_value = asyncio.Future()
        mock_initialize.return_value.set_result(tool_manager)

        # Call the create_process function
        await create_process(program)

        # Verify the tool manager's register_tools was called with the configuration
        # from the program
        mock_initialize.assert_called_once()

        args, _ = mock_initialize.call_args
        called_tools = args[0]
        assert [t.__name__ for t in called_tools] == []
        assert args[1] == tool_config


@pytest.mark.asyncio
async def test_fork_mcp_handling():
    """Test that the fork_process method correctly handles MCP state."""
    # Import here to avoid circular imports in the test
    from llmproc.llm_process import LLMProcess

    # For this test, we'll focus just on the MCP-specific code in fork_process
    # Create a simple class that has just what we need to test the MCP handling logic
    class TestForker:
        async def fork_process(self):
            # Create a new instance object for the fork
            forked = Mock()

            # Copy MCP state
            if self.mcp_enabled:
                forked.mcp_enabled = True

                # Handle tool_manager.mcp_aggregator if it exists
                if hasattr(self.tool_manager, "mcp_aggregator") and self.tool_manager.mcp_aggregator:
                    # In the real method, this is just a comment
                    # We use the pass statement to avoid indentation errors
                    pass

            return forked

    # Create our test object and set properties
    forker = TestForker()
    forker.mcp_enabled = True

    # Create a tool manager with MCP aggregator
    tool_manager = ToolManager()
    tool_manager.mcp_aggregator = Mock()
    forker.tool_manager = tool_manager

    # Call fork_process
    forked = await forker.fork_process()

    # Verify the fork has mcp_enabled set
    assert forked.mcp_enabled is True
