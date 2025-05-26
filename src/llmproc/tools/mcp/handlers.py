"""Handler functions for MCP tools.

This module provides functions for registering and handling MCP tools.
"""

import logging

# Use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Any, Optional

from llmproc.common.results import ToolResult
from llmproc.tools.mcp.constants import MCP_TOOL_SEPARATOR

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool

    from llmproc.mcp_registry import MCPAggregator
    from llmproc.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


def format_tool_for_anthropic(tool: "MCPTool", server_name: str | None = None) -> dict[str, Any]:
    """Format a tool for Anthropic API.

    Creates a properly formatted tool definition that can be used with Anthropic API.
    Applies proper namespacing with server_name and ensures the input schema has all
    required fields.

    Args:
        tool: Tool object from MCP registry
        server_name: Optional server name for proper namespacing

    Returns:
        Dictionary with tool information formatted for Anthropic API
    """
    # Create namespaced name with server prefix
    namespaced_name = f"{server_name}{MCP_TOOL_SEPARATOR}{tool.name}" if server_name else tool.name

    # Ensure input schema has required fields
    input_schema = tool.inputSchema.copy() if tool.inputSchema else {}
    if "type" not in input_schema:
        input_schema["type"] = "object"
    if "properties" not in input_schema:
        input_schema["properties"] = {}

    # Create the tool definition using the standard Anthropic API format
    return {
        "name": namespaced_name,
        "description": tool.description,
        "input_schema": input_schema,
    }


async def create_mcp_handler(
    tool: "MCPTool",
    server_name: str,
    tool_registry: "ToolRegistry",
    aggregator: "MCPAggregator",
    registered_tools: Optional[set[str]] = None,
) -> None:
    """Register an MCP tool with the tool registry.

    This function creates a handler function for the tool and registers it
    with the provided tool registry. The handler captures the aggregator
    in a closure, which is simpler and more efficient than using context-awareness
    for MCP tools which only need the static aggregator.

    Args:
        tool: The MCP tool to register
        server_name: The name of the server this tool belongs to
        tool_registry: The tool registry to register the tool with
        aggregator: The MCP aggregator to use for calling the tool
        registered_tools: Optional set to track registered tool names
    """
    # Initialize registered_tools if not provided
    if registered_tools is None:
        registered_tools = set()

    # Construct the full tool name
    full_tool_name = f"{server_name}{MCP_TOOL_SEPARATOR}{tool.name}"

    # Skip if the tool has already been registered
    if full_tool_name in registered_tools:
        logger.debug(f"Tool {full_tool_name} already registered, skipping")
        return

    # Format the tool for the LLM API
    tool_def = format_tool_for_anthropic(tool, server_name)

    # Create the handler function with explicit parameters only
    async def mcp_handler(**kwargs) -> ToolResult:
        """MCP tool handler function with explicit parameters.

        The handler captures the aggregator in closure, which is simpler and
        more efficient than context-awareness for MCP tools that only need
        access to a static aggregator instance.
        """
        try:
            # Call the tool via the aggregator using explicit parameters
            result = await aggregator.call_tool(server_name, tool.name, kwargs)

            # Check if the result is an error
            if result.isError:
                return ToolResult(content=result.content, is_error=True)

            # Return the successful result
            return ToolResult(content=result.content, is_error=False)

        except Exception as e:
            # Handle any exceptions that occur during tool execution
            error_message = f"Error calling MCP tool {full_tool_name}: {str(e)}"
            logger.error(error_message)
            return ToolResult.from_error(error_message)

    # Register the tool with the registry
    tool_registry.register_tool(full_tool_name, mcp_handler, tool_def)

    # Add to tool manager's enabled_tools if it has a tool_manager attribute
    if hasattr(tool_registry, "tool_manager"):
        if full_tool_name not in tool_registry.tool_manager.enabled_tools:
            tool_registry.tool_manager.enabled_tools.append(full_tool_name)
            logger.debug(f"Added MCP tool to enabled tools: {full_tool_name}")

    # Add the tool name to the set of registered tools
    registered_tools.add(full_tool_name)
