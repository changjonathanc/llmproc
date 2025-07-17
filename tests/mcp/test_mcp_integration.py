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
    assert MCPServerTools(server="server", default_access="read").default_access.value == "read"
    assert MCPServerTools(server="server", tools=["tool1"], default_access="admin").default_access.value == "admin"

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
@patch("llmproc.tools.mcp.MCPAggregator.initialize", return_value=[])
async def test_mcptool_descriptors(mock_initialize, mock_anthropic, mock_env, time_mcp_config):
    """Test program configuration with ``MCPServerTools`` descriptors."""
    # Setup mocks
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    from llmproc.tools.mcp import MCPServerTools

    # Create a program with MCPServerTools descriptors
    program = LLMProgram(
        model_name="claude-3-5-sonnet",
        provider="anthropic",
        system_prompt="You are an assistant with access to tools.",
        mcp_config_path=time_mcp_config,
        tools=[MCPServerTools(server="time", tools=["current"])],  # Using MCPServerTools descriptor
    )

    # Verify that the MCPServerTools descriptor was stored in the program config
    mcp_tools = [t for t in program.tools if isinstance(t, MCPServerTools)]
    assert len(mcp_tools) == 1
    assert mcp_tools[0].server == "time"
    assert mcp_tools[0].tools == ["current"]

    # Create a process
    process = await program.start()

    # Verify the MCPAggregator was initialized
    assert process.tool_manager.mcp_aggregator is not None
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
    mcp_tools2 = [t for t in program2.tools if isinstance(t, MCPServerTools)]
    assert len(mcp_tools2) == 1
    assert mcp_tools2[0].server == "time"
    assert mcp_tools2[0].tools == "all"

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
    mcp_tools3 = [t for t in program3.tools if isinstance(t, MCPServerTools)]
    assert len(mcp_tools3) == 2
    assert {d.server for d in mcp_tools3} == {"time", "calculator"}
