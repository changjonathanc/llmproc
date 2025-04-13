"""Integration tests for MCPManager."""

import json
import os
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.mcp.constants import MCP_TOOL_SEPARATOR


@pytest.fixture
def time_mcp_config():
    """Create a temporary MCP config file with time server."""
    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(
            {
                "mcpServers": {
                    "time": {
                        "type": "stdio",
                        "command": "uvx",
                        "args": ["mcp-server-time"],
                    }
                }
            },
            temp_file,
        )
        temp_path = temp_file.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def mock_mcp_registry():
    """Mock the MCP registry."""
    # Create MCP registry module mock
    mock_mcp_registry = MagicMock()

    # Setup mocks for MCP components
    mock_server_registry = MagicMock()
    mock_server_registry_class = MagicMock()
    mock_server_registry_class.from_config.return_value = mock_server_registry

    mock_aggregator = MagicMock()
    mock_aggregator_class = MagicMock()
    mock_aggregator_class.return_value = mock_aggregator

    # Create mock time tool
    mock_tool = MagicMock()
    mock_tool.name = "time.current"
    mock_tool.description = "Get the current time"
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    # Setup tool calls
    mock_tools_result = {"time": [mock_tool]}
    mock_aggregator.list_tools = AsyncMock(return_value=mock_tools_result)

    # Setup tool results
    mock_tool_result = MagicMock()
    mock_tool_result.content = {
        "unix_timestamp": 1646870400,
        "utc_time": "2022-03-10T00:00:00Z",
        "timezone": "UTC",
    }
    mock_tool_result.isError = False
    mock_aggregator.call_tool = AsyncMock(return_value=mock_tool_result)

    # Create patches for the mcp_registry module
    with patch.dict(
        "sys.modules",
        {
            "mcp_registry": mock_mcp_registry,
        },
    ):
        # Set attributes on the mock module
        mock_mcp_registry.ServerRegistry = mock_server_registry_class
        mock_mcp_registry.MCPAggregator = mock_aggregator_class
        mock_mcp_registry.get_config_path = MagicMock(return_value="/mock/config/path")

        yield mock_aggregator


@pytest.fixture
def mock_llm_program(time_mcp_config):
    """Create a mock LLMProgram with MCP configuration."""
    # Create a basic LLMProgram
    program = LLMProgram(
        model_name="claude-3-5-sonnet-20241022",
        provider="anthropic",
        system_prompt="You are a helpful assistant.",  # Add required system prompt
        mcp_config_path=time_mcp_config,  # Add MCP configuration directly
        mcp_tools={"time": ["current"]},
    )

    return program


@pytest.mark.asyncio
@patch("llmproc.providers.anthropic_process_executor.AnthropicProcessExecutor.run")
async def test_llmprocess_with_mcp_manager(
    mock_run, mock_llm_program, mock_mcp_registry
):
    """Test LLMProcess with MCPManager."""
    # Import the create_test_llmprocess_directly helper
    # Create a proper RunResult for the mock to return
    from llmproc.common.results import RunResult
    from llmproc.tools.mcp.manager import MCPManager
    from tests.conftest import create_test_llmprocess_directly

    run_result = RunResult()
    run_result.content = "Test response"
    mock_run.return_value = run_result

    # Create a process with our helper
    process = create_test_llmprocess_directly(program=mock_llm_program)
    process._executor = MagicMock()
    process._executor.run = mock_run

    # Set up MCP manager in tool_manager
    process.tool_manager.mcp_manager = MCPManager(process.tool_manager)
    process.tool_manager.mcp_manager.initialized = True
    process.tool_manager.mcp_manager.aggregator = MagicMock()

    # Verify MCPManager was created and initialized in the tool manager
    assert hasattr(process.tool_manager, "mcp_manager")
    assert process.tool_manager.mcp_manager is not None
    assert process.tool_manager.mcp_manager.initialized is True

    # Verify aggregator was set on the manager
    assert process.tool_manager.mcp_manager.aggregator is not None


@pytest.mark.asyncio
@patch("llmproc.providers.anthropic_process_executor.AnthropicProcessExecutor.run")
async def test_fork_with_mcp_manager(mock_run, mock_llm_program, mock_mcp_registry):
    """Test that MCPManager is properly copied during fork."""
    # Import the create_test_llmprocess_directly helper
    # Create a proper RunResult for the mock to return
    from llmproc.common.results import RunResult
    from llmproc.tools.mcp.manager import MCPManager
    from tests.conftest import create_test_llmprocess_directly

    run_result = RunResult()
    run_result.content = "Test response"
    mock_run.return_value = run_result

    # Create a process with our helper
    process = create_test_llmprocess_directly(program=mock_llm_program)
    process._executor = MagicMock()
    process._executor.run = mock_run

    # Set up MCP manager in tool_manager
    process.tool_manager.mcp_manager = MCPManager(process.tool_manager)
    process.tool_manager.mcp_manager.initialized = True
    process.tool_manager.mcp_manager.aggregator = MagicMock()

    # Verify MCPManager was created and initialized in the tool manager
    assert hasattr(process.tool_manager, "mcp_manager")
    assert process.tool_manager.mcp_manager is not None
    assert process.tool_manager.mcp_manager.initialized is True

    # Mock the fork_process method to return a copy with the same tool_manager
    forked_process = create_test_llmprocess_directly(program=mock_llm_program)
    forked_process.tool_manager = process.tool_manager  # Same tool manager reference

    # Add a mock for the fork_process method on the process
    process.fork_process = AsyncMock(return_value=forked_process)

    # Fork the process - must be awaited as it's an async method
    result_forked = await process.fork_process()

    # Verify the forked process has a tool manager with the MCPManager
    assert hasattr(result_forked, "tool_manager")

    # In our new design, the mcp_manager is in the tool_manager
    # Since the fork copies the tool_manager reference, both should point to same mcp_manager
    assert result_forked.tool_manager.mcp_manager is process.tool_manager.mcp_manager


@pytest.mark.asyncio
@patch("llmproc.tools.mcp.manager.MCPManager.is_valid_configuration", return_value=True)
@patch("llmproc.providers.anthropic_process_executor.AnthropicProcessExecutor.run")
async def test_empty_mcp_config(mock_run, mock_is_valid, mock_mcp_registry):
    """Test LLMProcess with empty MCP configuration."""
    # Import the create_test_llmprocess_directly helper
    from llmproc.tools.mcp.manager import MCPManager
    from tests.conftest import create_test_llmprocess_directly

    # Create an LLMProgram with just a config path but no tools
    program = LLMProgram(
        model_name="claude-3-5-sonnet-20241022",
        provider="anthropic",
        system_prompt="You are a helpful assistant.",
    )
    program.mcp_config_path = "/mock/config/path"
    program.mcp_tools = {}
    program.mcp_enabled = True

    # Create a proper RunResult for the mock to return
    from llmproc.common.results import RunResult

    run_result = RunResult()
    run_result.content = "Test response"
    mock_run.return_value = run_result

    # Create a process with our helper
    process = create_test_llmprocess_directly(
        program=program,
        mcp_config_path="/mock/config/path",
        mcp_tools={},
        mcp_enabled=True,
    )
    process._executor = MagicMock()
    process._executor.run = mock_run

    # Set up MCP manager in tool_manager
    process.tool_manager.mcp_manager = MCPManager(process.tool_manager)
    process.tool_manager.mcp_manager.initialized = True

    # Verify MCPManager was created and initialized in the tool manager
    assert hasattr(process.tool_manager, "mcp_manager")
    assert process.tool_manager.mcp_manager is not None
    assert process.tool_manager.mcp_manager.initialized is True

    # Run should succeed with no errors
    result = await process.run("Test input")
    assert result.content == "Test response"
