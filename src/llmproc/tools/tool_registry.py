"""Tool Registry for LLMProcess.

This module provides the ToolRegistry class which manages the registration,
access, and execution of tools for LLMProcess.
"""

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from llmproc.common.results import ToolResult

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
        self.tool_aliases = {}  # Mapping of alias names to actual tool names
        
    def clear(self):
        """Clear all registered tools and aliases.
        
        Returns:
            self (for method chaining)
        """
        self.tool_handlers.clear()
        self.tool_definitions.clear()
        self.tool_aliases.clear()
        
        logger.debug("Tool registry cleared.")
        return self
        
    def clear_non_mcp_tools(self):
        """Clear only non-MCP tools from the registry.
        
        This allows for selective clearing while preserving MCP tools,
        which might be managed separately.
        
        Returns:
            self (for method chaining)
        """
        from llmproc.tools.mcp.constants import MCP_TOOL_SEPARATOR
        
        # Identify MCP tools by their name format (containing separator)
        mcp_tool_names = [name for name in self.tool_handlers 
                         if MCP_TOOL_SEPARATOR in name]
                         
        # Keep MCP tool definitions
        mcp_definitions = [def_entry for def_entry in self.tool_definitions
                          if def_entry.get("name", "") in mcp_tool_names]
                          
        # Clear everything except MCP tools
        self.tool_handlers = {name: handler for name, handler in self.tool_handlers.items()
                             if name in mcp_tool_names}
        self.tool_definitions = mcp_definitions
        
        logger.debug(f"Cleared non-MCP tools. Retained {len(mcp_tool_names)} MCP tools.")
        return self

    def register_tool(self, name: str, handler: ToolHandler, definition: ToolSchema) -> ToolSchema:
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
            raise ValueError(f"Tool '{name}' not found. Available tools: {available_tools}")
        return self.tool_handlers[name]

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            A copy of the list of registered tool names to prevent external modification
        """
        return list(self.tool_handlers.keys())
        
    def get_tool_names(self) -> list[str]:
        """Get list of registered tool names.
        
        Returns:
            A copy of the list of all registered tool names to prevent external modification
        """
        return list(self.tool_handlers.keys())
        
    def register_aliases(self, aliases: dict[str, str]) -> None:
        """Register aliases for tools.
        
        Args:
            aliases: Dictionary mapping alias names to tool names
        """
        # Check for alias collision with existing tool names
        for alias, target in aliases.items():
            if alias in self.tool_handlers:
                logger.warning(f"Alias '{alias}' conflicts with existing tool name - this may cause confusion")
        
        self.tool_aliases.update(aliases)
        if aliases:
            logger.debug(f"Registered {len(aliases)} tool aliases")
        
    def get_definitions(self) -> list[ToolSchema]:
        """Get all tool definitions for API calls.

        Returns:
            A copy of the list of tool schemas to prevent external modification
        """
        return self.tool_definitions.copy()

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Call a tool by name with the given arguments.

        Args:
            name: The name of the tool to call (or an alias)
            args: The arguments to pass to the tool

        Returns:
            The result of the tool execution or an error ToolResult
        """
        # Resolve alias if it exists, otherwise use the original name
        resolved_name = self.tool_aliases.get(name, name)
        
        # Then check if the tool exists to handle "tool not found" errors
        if resolved_name not in self.tool_handlers:
            # Tool not found error
            logger.warning(f"Tool not found error: Tool '{name}' (resolved to '{resolved_name}') not found")

            # Get list of available tools for the error message
            available_tools = self.list_tools()
            
            # Add aliases to the error message if there are any
            alias_info = ""
            if self.tool_aliases:
                alias_info = "\n\nAvailable aliases: " + ", ".join(f"{k} -> {v}" for k, v in self.tool_aliases.items())

            # Create a helpful error message
            formatted_msg = f"Error: Tool '{name}' not found.\n\nAvailable tools: {', '.join(available_tools)}{alias_info}\n\nPlease try again with one of the available tools."

            # Return as an error ToolResult instead of raising an exception
            return ToolResult.from_error(formatted_msg)

        # If the tool exists, try to execute it
        try:
            handler = self.tool_handlers[resolved_name]
            
            # Extract parameters from args dictionary using function signature
            sig = inspect.signature(handler)
            
            # Function uses explicit parameters
            kwargs = {}
            
            # Extract parameters from args dictionary
            
            # Process parameters based on function signature
            for param_name, param in sig.parameters.items():
                # Skip internal process context parameters if they're not provided
                if param_name == "runtime_context" and param_name not in args:
                    continue
                    
                # Extract parameter from the args dictionary if available
                if param_name in args:
                    kwargs[param_name] = args[param_name]
            
            # Parameters prepared for handler
            
            # Call with extracted parameters
            result = await handler(**kwargs)
            
            # If this was an aliased tool, add a note to the result for debugging
            if name != resolved_name and isinstance(result, ToolResult):
                # If the result is not a ToolResult, we don't modify it
                result.alias_info = {"alias": name, "resolved": resolved_name}
                
            return result
        except Exception as e:
            # Handle errors during tool execution
            # If this was an aliased tool, include that in the error message
            if name != resolved_name:
                error_msg = f"Error executing tool '{name}' (aliased to '{resolved_name}'): {str(e)}"
            else:
                error_msg = f"Error executing tool '{name}': {str(e)}"
                
            logger.error(error_msg)

            return ToolResult.from_error(error_msg)
