"""Core tests for the MCP (Model Context Protocol) functionality.

This file consolidates core MCP tests from:
- test_mcp_tools.py
- test_mcp_manager.py
- test_mcp_add_tool.py
"""

import json
import os
import sys
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.common.results import ToolResult
from llmproc.program import LLMProgram
from llmproc.tools.mcp.constants import MCP_TOOL_SEPARATOR
from llmproc.tools.mcp.manager import MCPManager
from llmproc.tools.tool_registry import ToolRegistry
from tests.conftest import create_test_llmprocess_directly


# Common fixtures
@pytest.fixture
def mock_env():
    """Mock environment variables."""
    original_env = os.environ.copy()
    os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
    os.environ["GITHUB_TOKEN"] = "test-github-token"
    yield
    os.environ.clear()
    os.environ.update(original_env)


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


# Reusable utility functions
async def dummy_handler(args):
    """Simple dummy handler for testing."""
    return ToolResult.from_success("Test result")


def mock_mcp_registry():
    """Create a mock MCP registry module."""
    mock_registry = MagicMock()

    # Mock the ServerRegistry class
    mock_server_registry = MagicMock()
    mock_server_instance = MagicMock()
    mock_server_registry.from_config.return_value = mock_server_instance
    mock_server_instance.filter_servers.return_value = mock_server_instance
    mock_registry.ServerRegistry = mock_server_registry

    # Mock the MCPAggregator class
    mock_aggregator_class = MagicMock()
    mock_aggregator = AsyncMock()
    mock_aggregator.list_tools = AsyncMock(return_value={})
    mock_aggregator_class.return_value = mock_aggregator
    mock_registry.MCPAggregator = mock_aggregator_class

    return mock_registry


@pytest.mark.asyncio
async def test_manager_initialization(time_mcp_config):
    """Test MCPManager initialization with different configurations."""
    # Test 1: Basic initialization with minimal configuration
    manager = MCPManager(config_path=time_mcp_config, tools_config={"time": ["current"]})

    # Verify initial state
    assert manager.config_path == time_mcp_config
    assert manager.tools_config == {"time": ["current"]}
    assert manager.aggregator is None
    assert manager.initialized is False
    assert manager.is_enabled() is True

    # Test 2: Empty configuration
    empty_manager = MCPManager()
    assert empty_manager.config_path is None
    assert empty_manager.tools_config == {}
    assert empty_manager.aggregator is None
    assert empty_manager.initialized is False
    assert empty_manager.is_enabled() is False
    assert empty_manager.is_valid_configuration() is False

    # Test 3: Configuration with "all" tools
    all_tools_manager = MCPManager(config_path=time_mcp_config, tools_config={"time": "all"})
    assert all_tools_manager.config_path == time_mcp_config
    assert all_tools_manager.tools_config == {"time": "all"}
    assert all_tools_manager.is_valid_configuration() is True


@pytest.mark.asyncio
async def test_manager_with_mocked_registry():
    """Test MCPManager with properly mocked registry."""
    # Setup - Create MCPManager with test configuration
    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(
            {
                "mcpServers": {
                    "test-server": {
                        "type": "stdio",
                        "command": "echo",
                        "args": ["mock"],
                    }
                }
            },
            tmp,
        )
        config_path = tmp.name

    try:
        # Create registry and manager for testing
        registry = ToolRegistry()
        registry.tool_manager = MagicMock()
        registry.tool_manager.enabled_tools = []

        # Setup standard mock with time server
        standard_mock = mock_mcp_registry()

        # Test with properly mocked registry
        # Mocking sys.modules is the key to properly mock imports
        with patch("sys.modules", {**sys.modules, "mcp_registry": standard_mock}):
            # Create a fresh manager for each test for clean state
            manager = MCPManager(
                config_path=config_path, tools_config={"test-server": ["test-tool"]}, provider="anthropic"
            )

            success = await manager.initialize(registry)
            assert success is True
            assert manager.initialized is True
            assert len(registry.tool_handlers) == 0  # No mocked tools were registered

    finally:
        os.unlink(config_path)


@pytest.mark.asyncio
async def test_manager_validation():
    """Test MCPManager validation and configuration checks."""
    # Create temporary config for testing
    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(
            {
                "mcpServers": {
                    "test-server": {
                        "type": "stdio",
                        "command": "echo",
                        "args": ["mock"],
                    }
                }
            },
            tmp,
        )
        config_path = tmp.name

    try:
        # Test missing config path
        manager = MCPManager(config_path=None)

        # Check validation methods
        assert manager.is_enabled() is False
        assert manager.is_valid_configuration() is False

        # Test valid configuration
        manager = MCPManager(config_path=config_path, tools_config={"test-server": ["tool"]})

        # Check validation methods
        assert manager.is_enabled() is True
        assert manager.is_valid_configuration() is True
    finally:
        os.unlink(config_path)
