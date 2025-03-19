"""Tools for LLMProcess.

This module provides system tools that can be used by LLMProcess instances.
It also provides a registry to retrieve tool handlers and schemas by name.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from . import mcp
from .calculator import calculator_tool, calculator_tool_def
from .fork import fork_tool, fork_tool_def
from .spawn import spawn_tool, spawn_tool_def
from .tool_result import ToolResult

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
            The result of the tool execution

        Raises:
            ValueError: If the tool is not found
        """
        handler = self.get_handler(name)
        return await handler(args)


# Registry of available system tools
_SYSTEM_TOOLS = {
    "spawn": (spawn_tool, spawn_tool_def),
    "fork": (fork_tool, fork_tool_def),
    "calculator": (calculator_tool, calculator_tool_def),
}


def get_tool(name: str) -> tuple[ToolHandler, ToolSchema]:
    """Get a tool handler and schema by name.

    Args:
        name: The name of the tool to retrieve

    Returns:
        A tuple of (handler, schema) for the requested tool

    Raises:
        ValueError: If the tool is not found
    """
    if name not in _SYSTEM_TOOLS:
        available_tools = ", ".join(_SYSTEM_TOOLS.keys())
        raise ValueError(f"Tool '{name}' not found. Available tools: {available_tools}")

    return _SYSTEM_TOOLS[name]


def list_available_tools() -> list[str]:
    """List all available system tools.

    Returns:
        A list of available tool names
    """
    return list(_SYSTEM_TOOLS.keys())


def register_system_tools(registry: ToolRegistry, process) -> None:
    """Register system tools based on enabled tools in the process.

    Args:
        registry: The ToolRegistry to register tools with
        process: The LLMProcess instance
    """
    enabled_tools = getattr(process, "enabled_tools", [])

    # Register spawn tool if enabled and process has linked programs
    if "spawn" in enabled_tools and getattr(process, "has_linked_programs", False):
        register_spawn_tool(registry, process)

    # Register fork tool if enabled
    if "fork" in enabled_tools:
        register_fork_tool(registry, process)

    # Register calculator tool if enabled
    if "calculator" in enabled_tools:
        register_calculator_tool(registry)


def register_spawn_tool(registry: ToolRegistry, process) -> None:
    """Register the spawn tool with a registry.

    Args:
        registry: The ToolRegistry to register the tool with
        process: The LLMProcess instance to bind to the tool
    """
    # Get tool definition
    api_tool_def = spawn_tool_def.copy()

    # Customize description with available programs if any
    if hasattr(process, "linked_programs") and process.linked_programs:
        available_programs = ", ".join(process.linked_programs.keys())
        api_tool_def["description"] += f"\n\nAvailable programs: {available_programs}"

    # Create a handler function that binds to the process
    async def spawn_handler(args):
        return await spawn_tool(
            program_name=args.get("program_name"),
            query=args.get("query"),
            llm_process=process,
        )

    # Register with the registry
    registry.register_tool("spawn", spawn_handler, api_tool_def)

    logger.info("Registered spawn tool for process")


def register_fork_tool(registry: ToolRegistry, process) -> None:
    """Register the fork tool with a registry.

    Args:
        registry: The ToolRegistry to register the tool with
        process: The LLMProcess instance to bind to the tool
    """
    # Get tool definition
    api_tool_def = fork_tool_def.copy()

    # The actual fork implementation is handled by the process executor
    # This is just a placeholder handler for the interface
    async def fork_handler(args):
        return ToolResult.from_error(
            "Direct calls to fork_tool are not supported. This should be handled by the process executor."
        )

    # Register with the registry
    registry.register_tool("fork", fork_handler, api_tool_def)

    logger.info("Registered fork tool for process")


def register_calculator_tool(registry: ToolRegistry) -> None:
    """Register the calculator tool with a registry.

    Args:
        registry: The ToolRegistry to register the tool with
    """
    # Get tool definition
    api_tool_def = calculator_tool_def.copy()

    # Create a handler function
    async def calculator_handler(args):
        expression = args.get("expression", "")
        precision = args.get("precision", 6)
        return await calculator_tool(expression, precision)

    # Register with the registry
    registry.register_tool("calculator", calculator_handler, api_tool_def)

    logger.info("Registered calculator tool")


__all__ = [
    "spawn_tool",
    "spawn_tool_def",
    "fork_tool",
    "fork_tool_def",
    "calculator_tool",
    "calculator_tool_def",
    "get_tool",
    "list_available_tools",
    "ToolSchema",
    "ToolHandler",
    "ToolRegistry",
    "register_system_tools",
    "register_spawn_tool",
    "register_fork_tool",
    "register_calculator_tool",
    "ToolResult",
    "mcp",
]
