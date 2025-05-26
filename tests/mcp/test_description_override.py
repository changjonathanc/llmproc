import pytest
from mcp.types import ListToolsResult, Tool

from llmproc.common.metadata import get_tool_meta
from llmproc.tools.mcp import MCPServerTools
from llmproc.config.tool import ToolConfig
from llmproc.tools.mcp.manager import MCPManager


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


class DummyAggregator:
    async def _get_or_create_client(self, server_name):
        return DummyClient()


def create_manager():
    descriptor = MCPServerTools(
        server="calc",
        tools=[ToolConfig(name="add", description="override", param_descriptions={"a": "desc"})],
    )
    manager = MCPManager(mcp_tools=[descriptor])
    manager.aggregator = DummyAggregator()
    manager.initialized = True
    return manager


@pytest.mark.asyncio
async def test_description_override():
    manager = create_manager()
    regs = await manager.get_tool_registrations()
    name, handler, schema = regs[0]
    assert schema["description"] == "override"
    assert get_tool_meta(handler).description == "override"
    assert schema["input_schema"]["properties"]["a"]["description"] == "desc"
    assert schema["input_schema"]["properties"]["b"]["description"] == "B number"
    assert get_tool_meta(handler).param_descriptions == {
        "a": "desc",
        "b": "B number",
    }
