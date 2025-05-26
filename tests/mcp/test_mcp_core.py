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

from llmproc.common.access_control import AccessLevel
from llmproc.common.results import ToolResult
from llmproc.program import LLMProgram
from llmproc.tools.mcp import MCPServerTools
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
    """Create mock objects for MCP registry."""
    # Mock ServerRegistry class and instance
    mock_server_registry = MagicMock()
    mock_server_instance = MagicMock()
    mock_server_registry.from_config = MagicMock(return_value=mock_server_instance)
    mock_server_instance.filter_servers = MagicMock(return_value=mock_server_instance)

    # Mock MCPAggregator class and instance
    mock_aggregator_class = MagicMock()
    mock_aggregator = AsyncMock()
    mock_aggregator.list_tools = AsyncMock(return_value={})
    mock_aggregator_class.return_value = mock_aggregator

    return mock_server_registry, mock_server_instance, mock_aggregator_class


@pytest.mark.asyncio
async def test_manager_initialization(time_mcp_config):
    """Test MCPManager initialization with different configurations."""
    # Test 1: Basic initialization with minimal configuration
    manager = MCPManager(config_path=time_mcp_config, mcp_tools=[MCPServerTools(server="time", tools=["current"])])

    # Verify initial state
    assert manager.config_path == time_mcp_config
    assert len(manager.mcp_tools) == 1
    assert manager.mcp_tools[0].server == "time"
    assert manager.mcp_tools[0].tools == ["current"]
    assert manager.aggregator is None
    assert manager.initialized is False
    assert manager.is_enabled() is True

    # Test 2: Empty configuration
    empty_manager = MCPManager()
    assert empty_manager.config_path is None
    assert empty_manager.mcp_tools == []
    assert empty_manager.aggregator is None
    assert empty_manager.initialized is False
    assert empty_manager.is_enabled() is False
    assert empty_manager.is_valid_configuration() is False

    # Test 3: Configuration with "all" tools
    all_tools_manager = MCPManager(config_path=time_mcp_config, mcp_tools=[MCPServerTools(server="time")])
    assert all_tools_manager.config_path == time_mcp_config
    assert len(all_tools_manager.mcp_tools) == 1
    assert all_tools_manager.mcp_tools[0].tools == "all"
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

        # Setup standard mocks with time server
        mock_server_registry, mock_server_instance, mock_aggregator_class = mock_mcp_registry()

        # Test with properly mocked registry
        # Use patch.object to intercept method calls
        with (
            patch("llmproc.mcp_registry.ServerRegistry.from_config", return_value=mock_server_instance),
            patch("llmproc.mcp_registry.MCPAggregator", mock_aggregator_class),
        ):
            # Create a fresh manager for each test for clean state
            manager = MCPManager(
                config_path=config_path,
                mcp_tools=[MCPServerTools(server="test-server", tools=["test-tool"])],
                provider="anthropic",
            )

            success = await manager.initialize(registry)
            assert success is True
            assert manager.initialized is True
            assert len(registry.tool_handlers) == 0  # No mocked tools were registered

            # Ensure server filtering was performed
            mock_server_instance.filter_servers.assert_called_once_with(["test-server"])
            mock_aggregator_class.assert_called_once_with(mock_server_instance.filter_servers.return_value)

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
        manager = MCPManager(config_path=config_path, mcp_tools=[MCPServerTools(server="test-server", tools=["tool"])])

        # Check validation methods
        assert manager.is_enabled() is True
        assert manager.is_valid_configuration() is True
    finally:
        os.unlink(config_path)


def test_mcptoolsconfig_build_tools():
    """Test direct conversion from MCPToolsConfig to MCPServerTools."""
    from llmproc.config.mcp import MCPServerTools, MCPToolsConfig
    from llmproc.config.tool import ToolConfig

    # Create config with tool items
    config = MCPToolsConfig(root={"calc": [ToolConfig(name="add", access="read"), ToolConfig(name="sub")]})

    # Convert to server tools objects
    server_tools_list = config.build_mcp_tools()

    # Verify conversion
    assert len(server_tools_list) == 1
    server_tools = server_tools_list[0]
    assert isinstance(server_tools, MCPServerTools)
    assert server_tools.server == "calc"

    # Test tools conversion retains ToolConfig objects
    assert isinstance(server_tools.tools[0], ToolConfig)
    assert {t.name for t in server_tools.tools} == {"add", "sub"}

    # Test access level retrieval directly
    assert server_tools.get_access_level("add") == AccessLevel.READ
    assert server_tools.get_access_level("sub") == AccessLevel.WRITE


def test_program_loader_with_item_list(tmp_path):
    """ProgramLoader builds MCPServerTools objects from item lists."""
    from llmproc.config.mcp import MCPToolsConfig
    from llmproc.config.tool import ToolConfig
    from llmproc.config.program_loader import ProgramLoader
    from llmproc.config.schema import (
        LLMProgramConfig,
        MCPConfig,
        ModelConfig,
        PromptConfig,
        ToolsConfig,
    )

    mcp_json = tmp_path / "config.json"
    mcp_json.write_text("{}")

    config = LLMProgramConfig(
        model=ModelConfig(name="claude-3-5-sonnet", provider="anthropic"),
        prompt=PromptConfig(system_prompt="test"),
        mcp=MCPConfig(config_path=str(mcp_json)),
        tools=ToolsConfig(
            mcp=MCPToolsConfig(root={"calc": [ToolConfig(name="add", access="read"), ToolConfig(name="sub")]})
        ),
    )

    program = ProgramLoader._build_from_config(config, tmp_path)
    assert len(program.tool_manager.mcp_tools) == 1
    mcptool = program.tool_manager.mcp_tools[0]
    assert mcptool.server == "calc"
    assert {t.name for t in mcptool.tools} == {"add", "sub"}
    assert mcptool.get_access_level("add") == AccessLevel.READ
    assert mcptool.get_access_level("sub") == AccessLevel.WRITE

    # Test access levels (by extracting from both objects in a more flexible way)
    assert mcptool.get_access_level("add") == AccessLevel.READ
    assert mcptool.get_access_level("sub") == AccessLevel.WRITE
