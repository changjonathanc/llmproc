"""Tests for MCPManager class."""

import asyncio
import json
import os
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import constants directly
from llmproc.tools.mcp.constants import MCP_TOOL_SEPARATOR

# We'll import MCPManager inside the tests to avoid circular imports


@pytest.fixture
def mock_env():
    """Mock environment variables."""
    original_env = os.environ.copy()
    os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
    yield
    os.environ.clear()
    os.environ.update(original_env)


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


@pytest.mark.asyncio
async def test_mcp_manager_initialization(mock_mcp_registry, time_mcp_config, mock_env):
    """Test basic MCPManager initialization."""
    # Import here to avoid circular imports
    from llmproc.tools.mcp import MCPManager

    # Create manager instance
    manager = MCPManager(
        config_path=time_mcp_config, tools_config={"time": ["current"]}
    )

    # Verify initial state
    assert manager.config_path == time_mcp_config
    assert manager.tools_config == {"time": ["current"]}
    assert manager.aggregator is None
    assert manager.initialized is False

    # Verify is_enabled returns True
    assert manager.is_enabled() is True

    # Mock tool registry
    mock_tool_registry = MagicMock()
    mock_tool_registry.get_definitions.return_value = [{"name": "time__current"}]
    mock_tool_registry.tool_manager = MagicMock()
    mock_tool_registry.tool_manager.enabled_tools = []

    # Mock process
    mock_process = MagicMock()

    # Initialize the manager with updated signature (remove process param)
    success = await manager.initialize(mock_tool_registry)

    # Verify successful initialization
    assert success is True
    assert manager.initialized is True
    assert manager.aggregator is not None


@pytest.mark.asyncio
async def test_mcp_manager_empty_config(mock_mcp_registry):
    """Test MCPManager with empty configuration."""
    # Import here to avoid circular imports
    from llmproc.tools.mcp import MCPManager

    # Create manager instance with empty configuration
    manager = MCPManager()

    # Verify initial state
    assert manager.config_path is None
    assert manager.tools_config == {}
    assert manager.aggregator is None
    assert manager.initialized is False

    # Verify is_enabled returns False
    assert manager.is_enabled() is False

    # Verify is_valid_configuration returns False
    assert manager.is_valid_configuration() is False

    # Mock tool registry
    mock_tool_registry = MagicMock()

    # Mock process
    mock_process = MagicMock()

    # Initialize the manager - should return False because configuration is invalid
    success = await manager.initialize(mock_tool_registry)

    # Verify initialization was skipped
    assert success is False
    assert manager.initialized is False


@pytest.mark.asyncio
async def test_mcp_manager_with_all_tools(mock_mcp_registry, time_mcp_config, mock_env):
    """Test MCPManager with 'all' tools configuration."""
    # Import here to avoid circular imports
    from llmproc.tools.mcp import MCPManager

    # Create manager instance with "all" tools configuration
    manager = MCPManager(config_path=time_mcp_config, tools_config={"time": "all"})

    # Mock tool registry
    mock_tool_registry = MagicMock()
    mock_tool_registry.get_definitions.return_value = [{"name": "time__current"}]
    mock_tool_registry.tool_manager = MagicMock()
    mock_tool_registry.tool_manager.enabled_tools = []

    # Mock process
    mock_process = MagicMock()

    # Initialize the manager
    success = await manager.initialize(mock_tool_registry)

    # Verify successful initialization
    assert success is True
    assert manager.initialized is True

    # Verify tool filter has "all" configured correctly
    from mcp_registry import MCPAggregator

    MCPAggregator.assert_called_once()
    # The second argument to the constructor should be the tool_filter
    args, kwargs = MCPAggregator.call_args
    assert kwargs.get("tool_filter") == {"time": None}  # None means "all" tools


@pytest.mark.asyncio
async def test_mcp_manager_error_handling(mock_mcp_registry, time_mcp_config, mock_env):
    """Test MCPManager error handling during initialization."""
    # Import here to avoid circular imports
    from llmproc.tools.mcp import MCPManager

    # Create manager instance
    manager = MCPManager(
        config_path=time_mcp_config, tools_config={"time": ["current"]}
    )

    # Mock tool registry to return no tools
    mock_tool_registry = MagicMock()
    mock_tool_registry.get_definitions.return_value = []

    # Mock process
    mock_process = MagicMock()

    # Mock list_tools to return empty tools which will cause validation error
    mock_mcp_registry.list_tools = AsyncMock(return_value={})

    # Initialize the manager - should log a warning but not raise an error
    result = await manager.initialize(mock_tool_registry)

    # Initialization should succeed despite no tools being registered
    assert result is True


@pytest.mark.asyncio
@patch("llmproc.tools.mcp.manager.MCPManager.is_valid_configuration", return_value=True)
async def test_mcp_manager_no_servers(
    mock_is_valid, mock_mcp_registry, time_mcp_config, mock_env
):
    """Test MCPManager with no servers in config."""
    # Import here to avoid circular imports
    from llmproc.tools.mcp import MCPManager

    # Create manager instance with empty tools config
    manager = MCPManager(config_path=time_mcp_config, tools_config={})

    # Mock tool registry
    mock_tool_registry = MagicMock()

    # Mock process
    mock_process = MagicMock()

    # Initialize the manager
    success = await manager.initialize(mock_tool_registry)

    # Verify initialization was successful but no tools were registered
    assert success is True

    # Verify our mock is_valid_configuration was called
    mock_is_valid.assert_called_once()
