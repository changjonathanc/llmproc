"""Integration tests for MCP (Model Context Protocol) functionality with MCPTool descriptors.

This file tests the core MCP functionality using the new MCPTool descriptors approach.
"""

import json
import os
import tempfile
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.program import LLMProgram
from llmproc.tools.mcp import MCPTool
from llmproc.tools.mcp.constants import MCP_TOOL_SEPARATOR


def test_mcptool_descriptor_validation():
    """Test validation logic for MCPTool descriptors."""
    # Valid cases
    assert MCPTool(server="server").names == "all"
    assert MCPTool(server="server", names="tool1").names == ["tool1"]
    assert MCPTool(server="server", names=["tool1", "tool2"]).names == ["tool1", "tool2"]
    
    # Access level tests
    assert MCPTool(server="server", access="read").default_access.value == "read"
    assert MCPTool(server="server", names=["tool1"], access="admin").default_access.value == "admin"
    
    # Dictionary form
    tool_dict = MCPTool(server="server", names={"tool1": "read", "tool2": "write"})
    assert tool_dict.names == ["tool1", "tool2"]
    assert tool_dict.names_to_access["tool1"].value == "read"
    assert tool_dict.names_to_access["tool2"].value == "write"

    # Representation tests
    assert str(MCPTool(server="server")) == "<MCPTool server=ALL>"
    assert "tool1" in str(MCPTool(server="server", names="tool1"))

    # Invalid cases
    with pytest.raises(ValueError, match="non-empty server name"):
        MCPTool(server="")  # Empty server name

    with pytest.raises(ValueError, match="invalid tool names"):
        MCPTool(server="server", names=[""])  # Empty tool name

    with pytest.raises(ValueError, match="Invalid names type"):
        MCPTool(server="server", names=123)  # Invalid tool name type
        
    with pytest.raises(ValueError, match="Cannot specify both names dictionary and access parameter"):
        MCPTool(server="server", names={"tool1": "read"}, access="write")  # Conflicting access specifications

    with pytest.raises(ValueError, match="invalid tool names"):
        MCPTool(server="server", names=["valid", ""])  # Mix of valid and invalid tool names


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
                        "command": "echo",
                        "args": ["mock"],
                    }
                }
            },
            temp_file,
        )
        config_path = temp_file.name
    yield config_path
    os.unlink(config_path)


@pytest.mark.asyncio
@patch("llmproc.providers.providers.AsyncAnthropic")
@patch("llmproc.tools.mcp.manager.MCPManager.initialize")
async def test_mcptool_descriptors(mock_initialize, mock_anthropic, mock_env, time_mcp_config):
    """Test program configuration with MCPTool descriptors."""
    # Setup mocks
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_initialize.return_value = True

    # Create a program with MCPTool descriptors
    program = LLMProgram(
        model_name="claude-3-5-sonnet",
        provider="anthropic",
        system_prompt="You are an assistant with access to tools.",
        mcp_config_path=time_mcp_config,
        tools=[MCPTool(server="time", names=["current"])],  # Using MCPTool descriptor
    )

    # Verify that the MCPTool descriptor was stored in the tool_manager
    assert len(program.tool_manager.mcp_tools) == 1
    assert program.tool_manager.mcp_tools[0].server == "time"
    assert program.tool_manager.mcp_tools[0].names == ["current"]

    # Create a process
    process = await program.start()

    # Verify the MCPManager is initialized with the config path
    assert process.tool_manager.mcp_manager.config_path == time_mcp_config
    assert process.tool_manager.mcp_manager.config_path == time_mcp_config

    # Verify initialize was called
    mock_initialize.assert_called_once()

    # Test with 'all' tools
    program2 = LLMProgram(
        model_name="claude-3-5-sonnet",
        provider="anthropic",
        system_prompt="You are an assistant with access to tools.",
        mcp_config_path=time_mcp_config,
        tools=[MCPTool(server="time")],  # Using MCPTool descriptor with "all" tools
    )

    # Verify the descriptor was stored correctly with "all"
    assert len(program2.tool_manager.mcp_tools) == 1
    assert program2.tool_manager.mcp_tools[0].server == "time"
    assert program2.tool_manager.mcp_tools[0].names == "all"

    # Test with multiple MCPTool descriptors
    program3 = LLMProgram(
        model_name="claude-3-5-sonnet",
        provider="anthropic",
        system_prompt="You are an assistant with access to tools.",
        mcp_config_path=time_mcp_config,
        tools=[
            MCPTool(server="time", names=["current"]),
            MCPTool(server="calculator", names=["add", "subtract"]),
        ],
    )

    # Verify multiple descriptors are stored correctly
    assert len(program3.tool_manager.mcp_tools) == 2
    assert {d.server for d in program3.tool_manager.mcp_tools} == {"time", "calculator"}
