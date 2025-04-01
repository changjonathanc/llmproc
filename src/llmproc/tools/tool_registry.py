"""Tool Registry for LLMProcess.

This module provides the ToolRegistry class which manages the registration,
access, and execution of tools for LLMProcess.
"""

import logging
from typing import Any, Awaitable, Callable, Dict, List, TypedDict

from llmproc.tools.tool_result import ToolResult

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
    """Central registry for managing tools and their handlers.

    This class provides a unified interface for registering, accessing, and
    managing tools from different sources (system tools, MCP tools, etc.).
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self.tool_definitions = []  # Tool schemas for API calls
        self.tool_handlers = {}  # Mapping of tool names to handler functions

    def register_tool(
        self, name: str, handler: ToolHandler, definition: ToolSchema
    ) -> ToolSchema:
        """Register a tool with its handler and definition.

        Args:
            name: The name of the tool
            handler: The async function that handles tool calls
            definition: The tool schema/definition

        Returns:
            A copy of the tool definition that was registered
        """
        self.tool_handlers[name] = handler
        definition_copy = definition.copy()

        # Ensure the name in the definition matches the registered name
        definition_copy["name"] = name

        self.tool_definitions.append(definition_copy)
        logger.debug(f"Registered tool: {name}")
        return definition_copy

    def get_handler(self, name: str) -> ToolHandler:
        """Get a handler by tool name.

        Args:
            name: The name of the tool

        Returns:
            The tool handler function

        Raises:
            ValueError: If the tool is not found
        """
        if name not in self.tool_handlers:
            available_tools = ", ".join(self.tool_handlers.keys())
            raise ValueError(
                f"Tool '{name}' not found. Available tools: {available_tools}"
            )
        return self.tool_handlers[name]

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            A list of registered tool names
        """
        return list(self.tool_handlers.keys())

    def get_definitions(self) -> list[ToolSchema]:
        """Get all tool definitions for API calls.

        Returns:
            A list of tool schemas
        """
        return self.tool_definitions

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Call a tool by name with the given arguments.

        Args:
            name: The name of the tool to call
            args: The arguments to pass to the tool

        Returns:
            The result of the tool execution or an error ToolResult
        """
        # First check if the tool exists to handle "tool not found" errors
        if name not in self.tool_handlers:
            # Tool not found error
            logger.warning(f"Tool not found error: Tool '{name}' not found")
            
            # Get list of available tools for the error message
            available_tools = self.list_tools()
            
            # Create a helpful error message
            formatted_msg = (
                f"Error: Tool '{name}' not found.\n\n"
                f"Available tools: {', '.join(available_tools)}\n\n"
                f"Please try again with one of the available tools."
            )
            
            # Return as an error ToolResult instead of raising an exception
            return ToolResult.from_error(formatted_msg)
            
        # If the tool exists, try to execute it
        try:
            handler = self.tool_handlers[name]
            return await handler(args)
        except Exception as e:
            # Handle errors during tool execution
            error_msg = f"Error executing tool '{name}': {str(e)}"
            logger.error(error_msg)
            
            return ToolResult.from_error(error_msg)