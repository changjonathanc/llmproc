"""MCP tooling package."""

from llmproc.config.mcp import MCPServerTools
from llmproc.config.tool import ToolConfig

from .aggregator import MCPAggregator, create_mcp_tool_handler
from .connection_manager import ConnectionManager
from .exceptions import (
    MCPConnectionsDisabledError,
    MCPError,
    MCPServerConnectionError,
    MCPToolsLoadingError,
)
from .namespaced_tool import NamespacedTool
from .server_registry import MCPServerSettings
from .tool_loader import ToolLoader

__all__ = [
    "MCPAggregator",
    "ConnectionManager",
    "ToolLoader",
    "NamespacedTool",
    "MCPServerSettings",
    "create_mcp_tool_handler",
    "MCPError",
    "MCPConnectionsDisabledError",
    "MCPServerConnectionError",
    "MCPToolsLoadingError",
    "MCPServerTools",
    "ToolConfig",
]
