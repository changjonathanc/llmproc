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
    from llmproc.config.program_loader import ProgramLoader
    from llmproc.config.schema import (
        LLMProgramConfig,
        MCPConfig,
        ModelConfig,
        PromptConfig,
        ToolsConfig,
    )
    from llmproc.config.tool import ToolConfig
    from llmproc.program import LLMProgram

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

    data = ProgramLoader._build_from_config(config, tmp_path)
    program = LLMProgram._from_config_data(data)
    from llmproc.tools.mcp import MCPServerTools

    mcp_tools = [t for t in program.tools if isinstance(t, MCPServerTools)]
    assert mcp_tools
    mcptool = mcp_tools[0]
    assert mcptool.server == "calc"
    assert {t.name for t in mcptool.tools} == {"add", "sub"}
    assert mcptool.get_access_level("add") == AccessLevel.READ
    assert mcptool.get_access_level("sub") == AccessLevel.WRITE

    # Test access levels (by extracting from both objects in a more flexible way)
    assert mcptool.get_access_level("add") == AccessLevel.READ
    assert mcptool.get_access_level("sub") == AccessLevel.WRITE
