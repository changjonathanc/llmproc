"""Integration tests for MCP (Model Context Protocol) functionality with
``MCPServerTools`` descriptors.

This file tests the core MCP functionality using the new ``MCPServerTools``
descriptor approach.
"""

import json
import os
import tempfile
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llmproc.program import LLMProgram
from llmproc.tools.mcp import MCPServerTools
from llmproc.tools.mcp.constants import MCP_TOOL_SEPARATOR


def test_mcptool_descriptor_validation():
    """Test validation logic for ``MCPServerTools`` descriptors."""
    # Valid cases
    assert MCPServerTools(server="server").tools == "all"
    assert MCPServerTools(server="server", tools="tool1").tools == ["tool1"]
    assert MCPServerTools(server="server", tools=["tool1", "tool2"]).tools == ["tool1", "tool2"]

    # Access level tests
    assert MCPServerTools(server="server", access="read").default_access.value == "read"
    assert MCPServerTools(server="server", tools=["tool1"], access="admin").default_access.value == "admin"

    # Dictionary form
    tool_dict = MCPServerTools(server="server", tools={"tool1": "read", "tool2": "write"})
    assert {t.name for t in tool_dict.tools} == {"tool1", "tool2"}
    assert tool_dict.get_access_level("tool1").value == "read"
    assert tool_dict.get_access_level("tool2").value == "write"

    # Representation tests
    assert "ALL" in str(MCPServerTools(server="server"))
    assert "tool1" in str(MCPServerTools(server="server", tools="tool1"))

    # Invalid cases
    with pytest.raises(ValueError, match="Server name cannot be empty"):
        MCPServerTools(server="")  # Empty server name

    with pytest.raises(ValueError, match="Tool names cannot be empty"):
        MCPServerTools(server="server", tools=[""])  # Empty tool name

    with pytest.raises(ValueError, match="Unsupported tools specification type"):
        MCPServerTools(server="server", tools=123)  # Invalid tool name type

    with pytest.raises(ValueError, match="Cannot specify both tools dictionary and access parameter"):
        MCPServerTools(server="server", tools={"tool1": "read"}, access="write")  # Conflicting access specifications

    with pytest.raises(ValueError, match="Tool names cannot be empty"):
        MCPServerTools(server="server", tools=["valid", ""])  # Mix of valid and invalid tool names


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
    """Test program configuration with ``MCPServerTools`` descriptors."""
    # Setup mocks
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_initialize.return_value = True

    # Create a program with MCPServerTools descriptors
    program = LLMProgram(
        model_name="claude-3-5-sonnet",
        provider="anthropic",
        system_prompt="You are an assistant with access to tools.",
        mcp_config_path=time_mcp_config,
        tools=[MCPServerTools(server="time", tools=["current"])],  # Using MCPServerTools descriptor
    )

    # Verify that the MCPServerTools descriptor was stored in the tool_manager
    assert len(program.tool_manager.mcp_tools) == 1
    assert program.tool_manager.mcp_tools[0].server == "time"
    assert program.tool_manager.mcp_tools[0].tools == ["current"]

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
        tools=[MCPServerTools(server="time")],  # Using MCPServerTools descriptor with "all" tools
    )

    # Verify the descriptor was stored correctly with "all"
    assert len(program2.tool_manager.mcp_tools) == 1
    assert program2.tool_manager.mcp_tools[0].server == "time"
    assert program2.tool_manager.mcp_tools[0].tools == "all"

    # Test with multiple MCPServerTools descriptors
    program3 = LLMProgram(
        model_name="claude-3-5-sonnet",
        provider="anthropic",
        system_prompt="You are an assistant with access to tools.",
        mcp_config_path=time_mcp_config,
        tools=[
            MCPServerTools(server="time", tools=["current"]),
            MCPServerTools(server="calculator", tools=["add", "subtract"]),
        ],
    )

    # Verify multiple descriptors are stored correctly
    assert len(program3.tool_manager.mcp_tools) == 2
    assert {d.server for d in program3.tool_manager.mcp_tools} == {"time", "calculator"}
