"""MCP (Model Context Protocol) tools package for llmproc.

This package provides functionality for managing MCP servers and tools.
"""

from llmproc.tools.mcp.constants import MCP_TOOL_SEPARATOR
from llmproc.tools.mcp.handlers import create_mcp_handler, format_tool_for_anthropic
from llmproc.tools.mcp.integration import (
    initialize_mcp_tools,
    register_mcp_tool,
    register_runtime_mcp_tools,
)
from llmproc.tools.mcp.manager import MCPManager

__all__ = [
    "MCPManager",
    "MCP_TOOL_SEPARATOR",
    "create_mcp_handler",
    "format_tool_for_anthropic",
    "initialize_mcp_tools",
    "register_mcp_tool",
    "register_runtime_mcp_tools",
]
