import asyncio
import logging
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


def test_call_tool_error_logging(caplog):
    """Test that MCP server errors are logged with detailed information."""
    tool_a = Tool(name="a", inputSchema={})

    # Test error with message
    error_result_with_message = CallToolResult(
        isError=True, message="Test error message", content=[TextContent(type="text", text="Error details")]
    )

    # Test error without message
    error_result_without_message = CallToolResult(
        isError=True, message="", content=[TextContent(type="text", text="Some error content")]
    )

    # Test error with neither message nor content
    error_result_empty = CallToolResult(isError=True, message="", content=[])

    client1 = FakeClient(
        [tool_a], {"a": error_result_with_message, "b": error_result_without_message, "c": error_result_empty}
    )
    registry = FakeRegistry({"testserver": client1})
    aggregator = MCPAggregator(registry)

    # Test error with message
    with caplog.at_level(logging.ERROR):
        result = asyncio.run(aggregator.call_tool("testserver__a"))
        assert result.isError
        assert "MCP server 'testserver' returned error for tool 'a': Test error message" in caplog.text
        assert "Content: Error details" in caplog.text

    caplog.clear()

    # Test error without message but with content
    with caplog.at_level(logging.ERROR):
        result = asyncio.run(aggregator.call_tool("b", server_name="testserver"))
        assert result.isError
        assert "MCP server 'testserver' returned error for tool 'b'" in caplog.text
        assert "Content: Some error content" in caplog.text

    caplog.clear()

    # Test error with neither message nor content
    with caplog.at_level(logging.ERROR):
        result = asyncio.run(aggregator.call_tool("c", server_name="testserver"))
        assert result.isError
        assert "MCP server 'testserver' returned error for tool 'c'" in caplog.text
        assert "Available attributes:" in caplog.text
