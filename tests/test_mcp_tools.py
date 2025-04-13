"""Tests for MCP tool execution and error handling.

Note: We need an integration test for the tool manager and MCP tool interaction
to catch issues like the one fixed where MCP tools were registered but not enabled
in the tool manager's enabled_tools list.
"""

import asyncio
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc import LLMProcess
from llmproc.common.results import ToolResult
from llmproc.program import LLMProgram
from llmproc.tools.mcp.constants import MCP_TOOL_SEPARATOR
from llmproc.tools.mcp.manager import MCPManager
from tests.conftest import create_test_llmprocess_directly


@pytest.fixture
def mock_time_response():
    """Mock response for the time tool."""

    class ToolResponse:
        def __init__(self, time_data):
            self.content = time_data
            self.isError = False

    return ToolResponse(
        {
            "unix_timestamp": 1646870400,
            "utc_time": "2022-03-10T00:00:00Z",
            "timezone": "UTC",
        }
    )


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
def mock_env():
    """Mock environment variables."""
    original_env = os.environ.copy()
    os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_mcp_registry():
    """Mock the MCP registry with time tool."""
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
    mock_tools_result = MagicMock()
    mock_tools_result.tools = [mock_tool]
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


# Legacy test replaced by more appropriate error handling tests
@pytest.mark.skip("Test no longer relevant to current architecture")
@pytest.mark.asyncio
async def test_process_response_content():
    """Legacy test that is no longer needed with current architecture."""
    pass


@patch.dict("sys.modules", {"mcp_registry": MagicMock()})
@patch("llmproc.providers.providers.AsyncAnthropic")
@patch("asyncio.run")
def test_llm_process_with_time_tool(
    mock_asyncio_run, mock_anthropic, mock_env, time_mcp_config
):
    """Test LLMProcess with the time tool."""
    # Setup mock client
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    # Create program and process with MCP configuration
    from llmproc.program import LLMProgram

    program = LLMProgram(
        model_name="claude-3-5-haiku-20241022",
        provider="anthropic",
        system_prompt="You are an assistant with access to tools.",
        mcp_config_path=time_mcp_config,
        mcp_tools={"time": ["current"]},
    )

    # Use the proper pattern with mocked start() since this is a synchronous test
    with patch.object(program, "start") as mock_start:
        # Create mock process that would be returned by start()
        process = create_test_llmprocess_directly(program=program)

        # Set empty api_params to avoid None error
        process.api_params = {}

        # Set mcp_enabled for testing
        process.mcp_enabled = True

        # Configure mock to return our process
        mock_start.return_value = process

        # In a real implementation, we would use:
        # process = await program.start()

    # Check configuration
    assert process.mcp_tools == {"time": ["current"]}
    assert process.mcp_config_path == time_mcp_config
    assert process.mcp_tools == {"time": ["current"]}

    # In our new design, _initialize_tools no longer calls asyncio.run
    # Instead it's done lazily in run() or directly in create()
    # So we don't check mock_asyncio_run.assert_called_once()


@pytest.mark.asyncio
@patch.dict("sys.modules", {"mcp_registry": MagicMock()})
@patch("llmproc.providers.providers.AsyncAnthropic")
async def test_run_with_time_tool(mock_anthropic, mock_env, time_mcp_config):
    """Test the async run method with the time tool."""
    # Setup mock client
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    # Create program and process with MCP configuration
    from llmproc.program import LLMProgram

    with patch("llmproc.llm_process.asyncio.run"):
        program = LLMProgram(
            model_name="claude-3-5-haiku-20241022",
            provider="anthropic",
            system_prompt="You are an assistant with access to tools.",
            mcp_config_path=time_mcp_config,
            mcp_tools={"time": ["current"]},
        )

    # Use the proper pattern with AsyncMock since this is an async test
    mock_start = AsyncMock()
    program.start = mock_start

    # Create mock process that would be returned by start()
    process = create_test_llmprocess_directly(program=program)

    # Configure mock to return our process
    mock_start.return_value = process

    # In a real implementation, we would use:
    # process = await program.start()

    # Import RunResult from common.results to fix deprecation warning
    from llmproc.common.results import RunResult

    # Create a mock RunResult
    mock_run_result = RunResult()
    # Add a mock API call instead of setting api_calls directly
    mock_run_result.add_api_call({"model": "test-model"})

    # Patch the _async_run method directly to return the mock RunResult
    process._async_run = AsyncMock(return_value=mock_run_result)

    # Patch get_last_message to return our expected response
    process.get_last_message = MagicMock(
        return_value="The current time is 2022-03-10T00:00:00Z"
    )

    # Call the run method
    result = await process.run("What time is it now?")

    # Assert the result is our mock RunResult
    assert isinstance(result, RunResult)
    assert result.api_calls == 1

    # Check that the _async_run method was called
    process._async_run.assert_called_once_with("What time is it now?", 10, None)

    # In our new API design, get_last_message is not called inside the run method.
    # It's the responsibility of the caller to extract the message when needed.


@pytest.fixture
def mock_mcp_config():
    """Create a temporary MCP config file for testing."""
    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        config = {
            "mcpServers": {
                "existing-server": {
                    "type": "stdio",
                    "command": "/bin/echo",
                    "args": ["mock server"],
                }
            }
        }
        json.dump(config, tmp)
        tmp_path = tmp.name

    yield tmp_path
    os.unlink(tmp_path)


@pytest.mark.asyncio
@patch.dict("sys.modules", {"mcp_registry": MagicMock()})
@patch("llmproc.providers.providers.AsyncAnthropic")
@patch("mcp_registry.MCPAggregator")
@patch("mcp_registry.ServerRegistry")
async def test_unknown_server_error(
    mock_server_registry, mock_aggregator, mock_anthropic, mock_mcp_config
):
    """Test that MCPManager handles unknown servers gracefully."""
    # Setup mock client
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    # Mock the MCP registry module
    mock_mcp_registry = MagicMock()
    mock_mcp_registry.ServerRegistry = mock_server_registry
    mock_mcp_registry.MCPAggregator = mock_aggregator

    # Add necessary methods to the mocked mcp_registry module
    mock_mcp_registry.get_definitions = MagicMock()
    mock_mcp_registry.register_tool = MagicMock()

    # Setup mock instances
    mock_server_registry_instance = MagicMock()
    mock_server_registry.from_config.return_value = mock_server_registry_instance

    # Setup mock filter_servers to return the same mock registry
    mock_server_registry_instance.filter_servers = MagicMock(
        return_value=mock_server_registry_instance
    )

    mock_agg_instance = AsyncMock()
    mock_aggregator.return_value = mock_agg_instance

    # Mock list_tools to return a server_tools_map that doesn't include our target server
    mock_agg_instance.list_tools = AsyncMock(return_value={"existing-server": []})

    # Create a patch for the mcp_registry module
    with patch.dict("sys.modules", {"mcp_registry": mock_mcp_registry}):
        # Create program with non-existent server
        program = LLMProgram(
            model_name="claude-3-5-haiku-20241022",
            provider="anthropic",
            system_prompt="You are a helpful assistant with access to tools.",
            mcp_config_path=mock_mcp_config,
            mcp_tools={"non-existing-server": ["some-tool"]},
        )

        # The behavior has changed - we now log a warning instead of raising an error
        # Use program.start() instead of LLMProcess.create() for proper initialization
        process = await program.start()

        # Verify that the process was created successfully despite no tools being registered
        assert process is not None
        assert process.mcp_enabled is True


@pytest.mark.asyncio
@patch.dict("sys.modules", {"mcp_registry": MagicMock()})
@patch("llmproc.providers.providers.AsyncAnthropic")
@patch("mcp_registry.MCPAggregator")
@patch("mcp_registry.ServerRegistry")
async def test_unknown_tool_error(
    mock_server_registry, mock_aggregator, mock_anthropic, mock_mcp_config
):
    """Test that MCPManager handles unknown tools gracefully."""
    # Setup mock client
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    # Mock the MCP registry module
    mock_mcp_registry = MagicMock()
    mock_mcp_registry.ServerRegistry = mock_server_registry
    mock_mcp_registry.MCPAggregator = mock_aggregator

    # Add necessary methods to the mocked mcp_registry module
    mock_mcp_registry.get_definitions = MagicMock()
    mock_mcp_registry.register_tool = MagicMock()

    # Setup mock instances
    mock_server_registry_instance = MagicMock()
    mock_server_registry.from_config.return_value = mock_server_registry_instance

    # Setup mock filter_servers to return the same mock registry
    mock_server_registry_instance.filter_servers = MagicMock(
        return_value=mock_server_registry_instance
    )

    mock_agg_instance = AsyncMock()
    mock_aggregator.return_value = mock_agg_instance

    # Create mock tool
    mock_tool = MagicMock()
    mock_tool.name = "existing-tool"
    mock_tool.description = "A test tool"
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    # Mock list_tools to return a server_tools_map with our server but no tools
    # This will cause the MCP tool handler to not register any tools
    mock_agg_instance.list_tools = AsyncMock(return_value={"existing-server": []})

    # Create a patch for the mcp_registry module
    with patch.dict("sys.modules", {"mcp_registry": mock_mcp_registry}):
        # Create program with non-existent tool
        program = LLMProgram(
            model_name="claude-3-5-haiku-20241022",
            provider="anthropic",
            system_prompt="You are a helpful assistant with access to tools.",
            mcp_config_path=mock_mcp_config,
            mcp_tools={"existing-server": ["non-existing-tool"]},
        )

        # The behavior has changed - we now log a warning instead of raising an error
        # Use program.start() instead of LLMProcess.create() for proper initialization
        process = await program.start()

        # Verify that the process was created successfully despite no tools being registered
        assert process is not None
        assert process.mcp_enabled is True


@pytest.mark.asyncio
async def test_mcp_tools_in_tool_manager():
    """Test that MCP tools are properly registered in the tool manager.

    This test verifies that the integration functions correctly register MCP tools
    in the tool registry and update the enabled_tools list.
    """
    # Import the tools modules we want to test
    from llmproc.tools.mcp.integration import register_runtime_mcp_tools
    from llmproc.tools.tool_manager import ToolManager
    from llmproc.tools.tool_registry import ToolRegistry

    # Create a namespaced tool name
    namespaced_tool_name = f"test-server{MCP_TOOL_SEPARATOR}test-tool"

    # Create handler and schema for our test tool
    async def mock_handler(args):
        return ToolResult(content={"test": "result"}, is_error=False)

    tool_schema = {
        "name": namespaced_tool_name,
        "description": "Test tool description",
        "parameters": {"type": "object", "properties": {}},
    }

    # Create and set up registries
    mcp_registry = ToolRegistry()
    runtime_registry = ToolRegistry()

    # Create a tool manager that uses the runtime registry
    tool_manager = ToolManager()
    tool_manager.runtime_registry = runtime_registry
    runtime_registry.tool_manager = tool_manager

    # Register the MCP tool in the MCP registry
    mcp_registry.register_tool(namespaced_tool_name, mock_handler, tool_schema)

    # Create an initial list of enabled tools
    enabled_tools = ["some_other_tool"]
    tool_manager.enabled_tools = enabled_tools.copy()

    # Call the integration function to register MCP tools
    registered_count = register_runtime_mcp_tools(
        mcp_registry, runtime_registry, tool_manager.enabled_tools
    )

    # Verify one tool was registered
    assert registered_count == 1

    # Verify the tool is registered in the runtime registry
    assert namespaced_tool_name in runtime_registry.tool_handlers

    # Verify the tool was added to the enabled_tools list
    assert namespaced_tool_name in tool_manager.enabled_tools

    # Get tool definitions from the tool manager
    tool_schemas = tool_manager.get_tool_schemas()
    tool_names = [t.get("name") for t in tool_schemas]

    # Verify the tool appears in the schemas
    assert namespaced_tool_name in tool_names, "MCP tool not found in tool schemas"
