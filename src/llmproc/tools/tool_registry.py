"""Tool Registry for LLMProcess.

This module provides the ToolRegistry class which manages the registration,
access, and execution of tools for LLMProcess.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from llmproc.tools.core import Tool

# Set up logger
logger = logging.getLogger(__name__)


# Type definition for tool schemas
class ToolSchema(TypedDict):
    """Type definition for tool schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


# Type definition for tool handler
ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class ToolRegistry:
    """Central registry for managing :class:`Tool` instances.

    This class provides a unified interface for registering, accessing and
    executing tools from different sources (system tools, MCP tools, etc.).
    Internally it stores :class:`Tool` objects.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        # Map of tool name -> Tool object
        self._tools: dict[str, Tool] = {}

    def register_tool_obj(self, tool: Tool) -> bool:
        """Register a :class:`Tool` object directly.

        Args:
            tool: The :class:`Tool` instance to register

        Returns:
            True if registration succeeded, False if a tool with the same name already exists
        """
        name = tool.meta.name or tool.schema.get("name")
        if not name:
            raise ValueError("Tool object missing name")

        if name in self._tools:
            logger.debug("Tool '%s' already registered, skipping", name)
            return False

        # Ensure schema name matches the registry key
        tool.schema["name"] = name
        self._tools[name] = tool
        logger.debug("Registered tool: %s", name)
        return True

    def get_handler(self, name: str) -> ToolHandler:
        """Get a handler by tool name."""
        return self.get_tool(name).handler

    def get_tool(self, name: str) -> Tool:
        """Get a :class:`Tool` by name.

        Args:
            name: Tool name

        Returns:
            The :class:`Tool` instance

        Raises:
            ValueError: If the tool is not found
        """
        if name in self._tools:
            return self._tools[name]

        available = ", ".join(self._tools.keys())
        suggestion = "Run list_tools() to see available options." if available else "No tools registered."
        raise ValueError(f"Tool '{name}' not found. {suggestion} Available tools: {available}")

    def get_tool_names(self) -> list[str]:
        """Get list of registered tool names.

        Returns:
            A copy of the list of all registered tool names to prevent external modification
        """
        return list(self._tools.keys())

    def get_definitions(self) -> list[ToolSchema]:
        """Get all tool definitions for API calls.

        Returns:
            A copy of the list of tool schemas to prevent external modification
        """
        return [tool.schema.copy() for tool in self._tools.values()]
