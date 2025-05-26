"""MCP tool descriptor imports.

This module re-exports the configuration classes from ``llmproc.config.mcp``.
"""

# Import from mcp module
from llmproc.config.mcp import MCPServerTools
from llmproc.config.tool import ToolConfig

__all__ = ["MCPServerTools", "ToolConfig"]
