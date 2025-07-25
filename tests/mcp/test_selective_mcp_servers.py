"""Tests for selective MCP server initialization."""

import asyncio
import json
import os
import sys
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.tools.mcp import MCPServerTools
from llmproc.tools.tool_manager import ToolManager


def create_mock_mcp_registry():
    """Helper to create mocked MCP aggregator."""
    mock_aggregator_class = MagicMock()
    mock_aggregator = AsyncMock()
    mock_aggregator.list_tools = AsyncMock(return_value={})
    mock_aggregator_class.return_value = mock_aggregator
    return mock_aggregator_class


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
        mock_aggregator_class = create_mock_mcp_registry()
        mock_aggregator = mock_aggregator_class.return_value
        mock_aggregator.filter_servers = MagicMock(return_value=mock_aggregator)

        with patch("llmproc.tools.mcp.MCPAggregator.from_config", return_value=mock_aggregator):
            tm = ToolManager()
            await tm.register_tools(
                [MCPServerTools(server="s1")],
                {"mcp_enabled": True, "mcp_config_path": config_path},
            )

            mock_aggregator.filter_servers.assert_called_once_with(["s1"])
    finally:
        os.unlink(config_path)
