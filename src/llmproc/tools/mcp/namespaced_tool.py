"""Dataclass for namespaced MCP tools."""

from dataclasses import dataclass

from mcp.types import Tool as MCPTool


@dataclass
class NamespacedTool:
    """Metadata for a tool registered from an MCP server."""

    tool: MCPTool
    server_name: str
    original_name: str
