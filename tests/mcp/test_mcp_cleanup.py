"""Tests for MCPAggregator atexit cleanup behavior."""

import asyncio
import warnings
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from llmproc.mcp_registry.compound import MCPAggregator, MCPServerSettings, ServerRegistry
from mcp.types import ListToolsResult, Tool


class DummyClient:
    async def list_tools(self):
        return ListToolsResult(tools=[Tool(name="a", inputSchema={})])


class DummyRegistry(ServerRegistry):
    def __init__(self):
        super().__init__({"srv": MCPServerSettings()})
        self.client = DummyClient()

    def list_servers(self):
        return ["srv"]

    @asynccontextmanager
    async def get_client(self, server_name):
        yield self.client


def test_atexit_cleanup_no_cancel_warning(recwarn):
    """Ensure closing clients after loop shutdown does not warn."""
    registry = DummyRegistry()
    with patch("atexit.register") as mock_reg:
        aggregator = MCPAggregator(registry)
        close_func = mock_reg.call_args.args[0]

    loop = asyncio.new_event_loop()
    loop.run_until_complete(aggregator.list_tools())
    loop.close()

    warnings.simplefilter("always")
    close_func()
    assert not any("Attempted to exit cancel scope" in str(w.message) for w in recwarn.list)
