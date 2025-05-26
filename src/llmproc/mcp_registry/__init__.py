"""MCP Registry - A simplified MCP server aggregator and compound server implementation."""

from importlib.metadata import version

from .compound import MCPAggregator, MCPServerSettings, ServerRegistry

try:
    __version__ = version("llmproc")
except Exception:
    __version__ = "unknown"  # Fallback if package is not installed

__all__ = [
    "MCPServerSettings",
    "ServerRegistry",
    "MCPAggregator",
]
