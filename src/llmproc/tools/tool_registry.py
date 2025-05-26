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
        # Definitions and handlers for registered tools
        self.tool_definitions: list[ToolSchema] = []
        self.tool_handlers: dict[str, ToolHandler] = {}
        # Alias mappings
        # alias_to_real: alias name -> actual tool name
        self.tool_aliases: dict[str, str] = {}
        # real_to_alias: actual tool name -> alias name (one-to-one)
        self.real_to_alias: dict[str, str] = {}

    def register_tool(self, name: str, handler: ToolHandler, definition: ToolSchema) -> ToolSchema:
        """Register a tool with its handler and definition.

        Args:
            name: The name of the tool
            handler: The async function that handles tool calls
            definition: The tool schema/definition

        Returns:
            A copy of the tool definition that was registered
        """
        # Store handler with name as key
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
        # Resolve alias to real tool name
        real_name = self.tool_aliases.get(name, name)
        if real_name not in self.tool_handlers:
            available = ", ".join(self.tool_handlers.keys())
            raise ValueError(f"Tool '{name}' not found. Available tools: {available}")
        return self.tool_handlers[real_name]

    def get_tool_names(self) -> list[str]:
        """Get list of registered tool names.

        Returns:
            A copy of the list of all registered tool names to prevent external modification
        """
        return list(self.tool_handlers.keys())

    def register_aliases(self, aliases: dict[str, str]) -> None:
        """
        Register alias names for existing tools.

        Args:
            aliases: Mapping of alias -> actual tool name

        Stores both alias_to_real and rebuilds a reverse real_to_alias map
        for later schema aliasing and runtime resolution.
        """
        # Warn if alias shadows an existing tool name
        for alias, target in aliases.items():
            if alias in self.tool_handlers:
                logger.warning(f"Alias '{alias}' conflicts with existing tool name - may cause confusion")
        # Update alias -> real mapping
        self.tool_aliases.update(aliases)
        # Rebuild real -> alias map (one-to-one); last alias wins if conflicts
        self.real_to_alias = {real: alias for alias, real in self.tool_aliases.items()}
        if aliases:
            logger.debug(f"Registered {len(aliases)} tool aliases")

    def get_definitions(self) -> list[ToolSchema]:
        """Get all tool definitions for API calls.

        Returns:
            A copy of the list of tool schemas to prevent external modification
        """
        return self.tool_definitions.copy()

    def alias_schemas(self, schemas: list[ToolSchema]) -> list[ToolSchema]:
        """
        Apply alias names to a list of tool schemas.

        For each schema whose 'name' matches a real tool name with a configured alias,
        returns a copy with the 'name' field replaced by the alias.

        Args:
            schemas: List of tool schemas with real 'name' fields

        Returns:
            List of schemas with 'name' replaced by alias where configured
        """
        aliased: list[ToolSchema] = []
        for schema in schemas:
            real = schema.get("name", "")
            if real in self.real_to_alias:
                s2 = schema.copy()
                s2["name"] = self.real_to_alias[real]
                aliased.append(s2)
            else:
                aliased.append(schema)
        return aliased

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """
        Invoke a registered tool by name or alias.

        This method resolves the provided name through the alias mapping,
        calls the corresponding handler, and if an alias was used,
        stamps `alias_info` on the returned ToolResult for tracing.

        Args:
            name: The tool name or its configured alias
            args: Arguments to pass to the tool handler

        Returns:
            The result of the tool execution, potentially with alias_info,
            or an error ToolResult if lookup or execution fails.
        """
        # Resolve alias to real tool name for handler lookup
        resolved_name = self.tool_aliases.get(name, name)

        # Check if tool exists
        if resolved_name not in self.tool_handlers:
            # Log for debugging but keep message simple
            logger.warning(f"Tool not found: '{name}' (resolved to '{resolved_name}')")
            return ToolResult.from_error("This tool is not available")

        # Execute the tool with simple error handling
        try:
            handler = self.tool_handlers[resolved_name]
            result = await handler(**args)

            # If an alias was used, record alias info on the result for tracing
            if name != resolved_name and isinstance(result, ToolResult):
                result.alias_info = {"alias": name, "resolved": resolved_name}

            return result
        except Exception as e:
            # Log with full details for debugging
            logger.error(f"Error executing tool '{resolved_name}': {str(e)}", exc_info=True)
            # Pass through the exception message directly
            return ToolResult.from_error(f"Error: {str(e)}")
