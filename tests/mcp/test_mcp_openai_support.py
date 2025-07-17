"""Tests for OpenAI provider support in MCPAggregator."""

import pytest
from mcp.types import ListToolsResult, Tool
from unittest.mock import MagicMock

from llmproc.tools.mcp import MCPServerTools, MCPAggregator, NamespacedTool
from llmproc.providers.openai_utils import convert_tools_to_openai_format
from llmproc.tools.function_tools import create_handler_from_function


class DummyAggregator(MCPAggregator):
    def __init__(self):
        super().__init__(MagicMock())

    async def load_servers(self, specific_servers=None):
        tool = Tool(name="srv__echo", inputSchema={})
        self._namespaced_tools = {
            "srv__echo": NamespacedTool(
                tool=tool,
                server_name="srv",
                original_name="echo",
            )
        }


def create_aggregator():
    descriptor = MCPServerTools(server="srv")
    agg = DummyAggregator()
    agg.mcp_tools = [descriptor]
    agg.provider = "openai"
    return agg


@pytest.mark.asyncio
async def test_openai_formatting():
    """Validate OpenAI schema conversion for MCP tools."""
    agg = create_aggregator()
    regs = await agg.initialize(agg.mcp_tools)
    tool = regs[0]
    converted = convert_tools_to_openai_format([tool.schema])[0]
    assert converted["type"] == "function"
    assert converted["function"]["name"] == "srv__echo"
    assert "parameters" in converted["function"]
