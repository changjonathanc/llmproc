"""Tests for MCP description and parameter overrides."""

import pytest
from mcp.types import ListToolsResult, Tool
from unittest.mock import MagicMock

from llmproc.common.metadata import get_tool_meta
from llmproc.tools.mcp import MCPServerTools, MCPAggregator, NamespacedTool
from llmproc.config.tool import ToolConfig
from llmproc.tools.function_tools import create_handler_from_function


class DummyClient:
    async def list_tools(self):
        return ListToolsResult(
            tools=[
                Tool(
                    name="add",
                    description="orig",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer", "description": "A number"},
                            "b": {"type": "integer", "description": "B number"},
                        },
                        "required": ["a", "b"],
                    },
                )
            ]
        )



class DummyAggregator(MCPAggregator):
    def __init__(self):
        super().__init__(MagicMock())

    async def load_servers(self, specific_servers=None):
        tool = Tool(
            name="calc__add",
            description="orig",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "A number"},
                    "b": {"type": "integer", "description": "B number"},
                },
                "required": ["a", "b"],
            },
        )
        self._namespaced_tools = {
            "calc__add": NamespacedTool(
                tool=tool,
                server_name="calc",
                original_name="add",
            )
        }


def create_aggregator():
    descriptor = MCPServerTools(
        server="calc",
        tools=[ToolConfig(name="add", description="override", param_descriptions={"a": "desc"})],
    )
    agg = DummyAggregator()
    agg.mcp_tools = [descriptor]
    return agg


@pytest.mark.asyncio
async def test_description_override():
    """Verify MCP description and parameter overrides are applied."""
    agg = create_aggregator()
    regs = await agg.initialize(agg.mcp_tools)
    tool = regs[0]
    handler = tool.handler
    schema = tool.schema
    assert schema["description"] == "override"
    assert get_tool_meta(handler).description == "override"
    assert schema["input_schema"]["properties"]["a"]["description"] == "desc"
    assert schema["input_schema"]["properties"]["b"]["description"] == "B number"
    assert get_tool_meta(handler).param_descriptions == {
        "a": "desc",
        "b": "B number",
    }
