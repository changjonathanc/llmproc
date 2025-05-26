"""Tests for selective MCP server initialization."""

import asyncio
import json
import os
import sys
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llmproc.tools.mcp import MCPServerTools
from llmproc.tools.mcp.manager import MCPManager
from llmproc.tools.tool_registry import ToolRegistry


def create_mock_mcp_registry():
    """Helper to create mocked MCP registry objects."""
    # Mock ServerRegistry class and instance
    mock_server_registry = MagicMock()
    mock_server_instance = MagicMock()
    mock_server_registry.from_config = MagicMock(return_value=mock_server_instance)
    mock_server_instance.filter_servers = MagicMock(return_value=mock_server_instance)

    # Mock MCPAggregator class and instance
    mock_aggregator_class = MagicMock()
    mock_aggregator = AsyncMock()
    mock_aggregator.list_tools = AsyncMock(return_value={})
    mock_aggregator_class.return_value = mock_aggregator

    return mock_server_registry, mock_server_instance, mock_aggregator_class


@pytest.mark.asyncio
async def test_filter_servers_called():
    """Ensure only specified servers are initialized."""
    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(
            {
                "mcpServers": {
                    "s1": {"type": "stdio", "command": "echo", "args": ["m1"]},
                    "s2": {"type": "stdio", "command": "echo", "args": ["m2"]},
                }
            },
            tmp,
        )
        config_path = tmp.name

    try:
        registry = ToolRegistry()
        mock_server_registry, mock_server_instance, mock_aggregator_class = create_mock_mcp_registry()

        with (
            patch("llmproc.mcp_registry.ServerRegistry.from_config", return_value=mock_server_instance),
            patch("llmproc.mcp_registry.MCPAggregator", mock_aggregator_class),
        ):
            manager = MCPManager(
                config_path=config_path,
                mcp_tools=[MCPServerTools(server="s1")],
            )
            await manager.initialize(registry)

            mock_server_instance.filter_servers.assert_called_once_with(["s1"])
    finally:
        os.unlink(config_path)
