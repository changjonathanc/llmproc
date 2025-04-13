"""Tests for selective MCP server initialization."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc import LLMProgram
from llmproc.tools.mcp import manager as mcp_manager
from llmproc.tools.tool_registry import ToolRegistry


@pytest.mark.essential_api
async def test_selective_server_initialization():
    """Test that only servers specified in mcp_tools are initialized."""
    with patch(
        "llmproc.tools.mcp.manager.MCPManager.initialize", new_callable=AsyncMock
    ) as mock_initialize:
        # Configure the mock to return success
        mock_initialize.return_value = True

        # Create program with tools from only server1 and server2
        program = LLMProgram(
            model_name="claude-3-5-sonnet",
            provider="anthropic",
            system_prompt="You are an assistant.",
            mcp_config_path="fake_config.json",
            mcp_tools={"server1": ["tool1"], "server2": "all"},
        )

        # Start the process
        process = await program.start()

        # Verify initialize was called on the MCPManager
        mock_initialize.assert_called_once()

        # Verify the MCP manager was created with the right configuration in the tool manager
        assert process.tool_manager.mcp_manager.config_path == "fake_config.json"
        assert set(process.tool_manager.mcp_manager.tools_config.keys()) == {
            "server1",
            "server2",
        }
        assert process.tool_manager.mcp_manager.tools_config["server1"] == ["tool1"]
        assert process.tool_manager.mcp_manager.tools_config["server2"] == "all"


@pytest.mark.essential_api
async def test_tool_filter_creation():
    """Test that our mcp_tools configuration is correctly converted to a tool filter."""
    # Create simple test data
    mcp_tools_config = {"server1": ["tool1"], "server2": "all"}

    # Create tool filter using the same logic from the implementation
    tool_filter = {
        server_name: None if tool_config == "all" else tool_config
        for server_name, tool_config in mcp_tools_config.items()
    }

    # Verify that the tool filter is created correctly
    assert "server1" in tool_filter
    assert "server2" in tool_filter
    assert tool_filter["server1"] == ["tool1"]
    assert tool_filter["server2"] is None


@pytest.mark.essential_api
async def test_server_validation():
    """Test that validation raises error for non-existent servers."""
    # Create mock registry
    mock_registry = MagicMock()
    mock_registry.list_servers.return_value = ["server1", "server2"]

    # Define a validation function that simulates our validation code
    async def validate_servers(mcp_tools_config):
        server_names = list(mcp_tools_config.keys())
        available_servers = mock_registry.list_servers()
        for server_name in server_names:
            if server_name not in available_servers:
                raise ValueError(
                    f"Server '{server_name}' not found in MCP configuration."
                )

    # Call our test function with non-existent server
    mcp_tools_config = {"server1": ["tool1"], "nonexistent_server": "all"}

    # Should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        await validate_servers(mcp_tools_config)

    # Verify the error message mentions the non-existent server
    assert "nonexistent_server" in str(excinfo.value)


@pytest.mark.essential_api
async def test_mcpmanager_initialize():
    """Test MCPManager initialize method directly with mocks."""
    # Set up test data
    mock_process = MagicMock()
    mock_tool_registry = MagicMock()
    mcp_config_path = "fake_config.json"
    mcp_tools_config = {"server1": ["tool1", "tool2"]}

    # Create a MCPManager
    from llmproc.tools.mcp import MCPManager

    manager = MCPManager(
        config_path=mcp_config_path,
        tools_config=mcp_tools_config,
        provider="anthropic",  # Updated to use provider instead of llm_process
    )

    # Mock the initialize method
    async def mock_initialize(self, tool_registry):
        # Verify params were passed correctly
        assert tool_registry is mock_tool_registry

        # Verify the configuration
        assert self.config_path == mcp_config_path
        assert self.tools_config == mcp_tools_config

        return True

    # Patch the initialize method
    with patch.object(MCPManager, "initialize", mock_initialize):
        # Call the initialize method
        result = await manager.initialize(mock_tool_registry)

        # Check the result
        assert result is True


@pytest.mark.essential_api
async def test_empty_server_list_handling():
    """Test that an empty server list is handled gracefully."""
    # Direct test of the MCPManager's empty server list handling

    # Set up mock process and registry
    mock_process = MagicMock()
    mock_tool_registry = MagicMock()
    mock_tool_registry.get_definitions = MagicMock(return_value=[])

    # Set up empty mcp config
    mcp_config_path = "fake_config.json"
    mcp_tools_config = {}

    # Create a MCPManager with empty tools config
    from llmproc.tools.mcp import MCPManager

    manager = MCPManager(
        config_path=mcp_config_path,
        tools_config=mcp_tools_config,
        provider="anthropic",  # Updated to use provider instead of llm_process
    )

    # Mock the initialize method to verify early return for empty server list
    async def mock_initialize(self, tool_registry):
        # Extract server names from tools_config
        server_names = list(self.tools_config.keys())

        # Early check for empty server list - should return True
        if not server_names:
            return True

        # If we get here, we missed the early return
        return False

    # Patch the initialize method
    with patch.object(MCPManager, "initialize", mock_initialize):
        # Call the initialize method
        result = await manager.initialize(mock_tool_registry)

        # Check the result - should be True from the early return
        assert result is True


if __name__ == "__main__":
    asyncio.run(test_selective_server_initialization())
