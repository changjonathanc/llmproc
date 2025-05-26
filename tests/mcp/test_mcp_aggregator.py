import asyncio
from contextlib import asynccontextmanager

import pytest
from llmproc.mcp_registry.compound import MCPAggregator, MCPServerSettings, ServerRegistry
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool


class FakeClient:
    def __init__(self, tools, call_results):
        self._tools = tools
        self._call_results = call_results

    async def list_tools(self):
        return ListToolsResult(tools=self._tools)

    async def call_tool(self, name, arguments=None):
        return self._call_results[name]


class FakeRegistry(ServerRegistry):
    def __init__(self, clients):
        servers = {name: MCPServerSettings() for name in clients}
        super().__init__(servers)
        self.clients = clients

    def list_servers(self):
        return list(self.clients.keys())

    @asynccontextmanager
    async def get_client(self, server_name):
        yield self.clients[server_name]


def test_list_tools_namespaced():
    tool_a = Tool(name="a", inputSchema={})
    tool_b = Tool(name="b", inputSchema={})

    client1 = FakeClient(
        [tool_a], {"a": CallToolResult(isError=False, message="", content=[TextContent(type="text", text="A")])}
    )
    client2 = FakeClient(
        [tool_b], {"b": CallToolResult(isError=False, message="", content=[TextContent(type="text", text="B")])}
    )

    registry = FakeRegistry({"s1": client1, "s2": client2})
    aggregator = MCPAggregator(registry)

    result = asyncio.run(aggregator.list_tools())
    names = sorted(t.name for t in result.tools)
    assert names == ["s1__a", "s2__b"]


def test_call_tool_basic():
    tool_a = Tool(name="a", inputSchema={})
    call_result = CallToolResult(isError=False, message="", content=[TextContent(type="text", text="A result")])
    client1 = FakeClient([tool_a], {"a": call_result})
    registry = FakeRegistry({"s1": client1})
    aggregator = MCPAggregator(registry)

    result = asyncio.run(aggregator.call_tool("s1__a"))
    assert result.content[0].text == "A result"

    result2 = asyncio.run(aggregator.call_tool("a", server_name="s1"))
    assert result2.content[0].text == "A result"
