"""MCP tool integration for LLMProcess.

This module provides functions to register and manage MCP (Model Context Protocol) tools.
It handles the initialization, registration, and execution of MCP tools.
"""

import logging
from typing import Any

# Set up logger
logger = logging.getLogger(__name__)

# Define TypeAlias (available in Python 3.10+)
# If pre-3.10, we could use a simple variable
MCPTool = Any
MCPToolsMap = dict[str, list[Any]]


async def initialize_mcp_tools(
    process, tool_registry, mcp_config_path: str, mcp_tools_config: dict[str, Any]
) -> bool:
    """Initialize MCP registry and tools.

    This sets up the MCP registry and filters tools based on user configuration.
    Only tools explicitly specified in the mcp_tools configuration will be enabled.
    Only servers that have tools configured will be initialized.

    Args:
        process: The LLMProcess instance that owns these tools
        tool_registry: The ToolRegistry to register tools with
        mcp_config_path: Path to the MCP configuration file
        mcp_tools_config: Dictionary mapping server names to tool configurations

    Returns:
        True if initialization was successful, False otherwise

    Raises:
        RuntimeError: If MCP registry initialization or tool listing fails
        ValueError: If a server specified in mcp_tools_config is not found in available tools
    """
    try:
        # Import MCP registry here to avoid circular imports
        from mcp_registry import MCPAggregator, ServerRegistry

        # Initialize registry and aggregator
        registry = ServerRegistry.from_config(mcp_config_path)
        aggregator = process.aggregator = MCPAggregator(registry)

        # Get tools grouped by server
        server_tools_map = await aggregator.list_tools(return_server_mapping=True)

        # Track registered tools to avoid duplicates
        registered_tools = set()

        # Process each server and tool configuration
        for server_name, tool_config in mcp_tools_config.items():
            # Raise error when servers don't exist in the available tools
            if server_name not in server_tools_map:
                raise ValueError(f"Server '{server_name}' not found in MCP configuration. Check your MCP servers configuration file and ensure it's properly configured.")

            server_tools = server_tools_map[server_name]

            # Create a mapping of tool names to tools for this server
            server_tool_map = {tool.name: tool for tool in server_tools}

            # Case 1: Register all tools for a server
            if tool_config == "all":
                for tool in server_tools:
                    await register_mcp_tool(
                        tool, server_name, tool_registry, aggregator, registered_tools
                    )

                logger.info(
                    f"Registered all tools ({len(server_tools)}) from server '{server_name}'"
                )

            # Case 2: Register specific tools
            elif isinstance(tool_config, list):
                for tool_name in tool_config:
                    if tool_name in server_tool_map:
                        tool = server_tool_map[tool_name]
                        await register_mcp_tool(
                            tool,
                            server_name,
                            tool_registry,
                            aggregator,
                            registered_tools,
                        )
                    else:
                        raise ValueError(
                            f"Tool '{tool_name}' not found for server '{server_name}'. "
                            f"Available tools for this server are: {', '.join(server_tool_map.keys())}"
                        )

                logger.info(
                    f"Registered {len([t for t in tool_config if t in server_tool_map])} tools from server '{server_name}'"
                )

        # Summarize registered tools
        if not tool_registry.get_definitions():
            # If there are mcp_tools_config entries but no tools registered, this is an error
            if mcp_tools_config:
                raise ValueError(
                    "No MCP tools were registered despite configuration being provided. "
                    "Check that the server names and tool names in your configuration are correct."
                )
            else:
                logger.warning("No MCP tools were registered. Check your configuration.")
        else:
            logger.info(
                f"Total MCP tools registered: {len(tool_registry.get_definitions())}"
            )

        return True

    except ValueError as e:
        # Let validation errors propagate up to the caller
        logger.error(f"Failed to initialize MCP tools: {str(e)}")
        raise  # Re-raise the original ValueError
    except Exception as e:
        # Log and return False for other types of errors
        logger.error(f"Failed to initialize MCP tools: {str(e)}")
        return False


async def register_mcp_tool(
    tool: MCPTool,
    server_name: str,
    tool_registry,
    aggregator,
    registered_tools: set[str],
) -> None:
    """Register an MCP tool with the tool registry.

    Args:
        tool: The MCP tool object to register
        server_name: The name of the server the tool belongs to
        tool_registry: The ToolRegistry to register tools with
        aggregator: The MCP Aggregator that will call the tools
        registered_tools: Set of already registered tool names to avoid duplicates
    """
    namespaced_name = f"{server_name}__{tool.name}"
    if namespaced_name not in registered_tools:
        # Format the tool for Anthropic API
        tool_def = format_tool_for_anthropic(tool, server_name)

        # Create a handler for this tool
        async def mcp_tool_handler(args):
            try:
                # Call the tool through the aggregator
                result = await aggregator.call_tool(namespaced_name, args)
                # Return a standardized ToolResult
                from llmproc.tools.tool_result import ToolResult

                return ToolResult(content=result.content, is_error=result.isError)
            except Exception as e:
                error_msg = f"Error executing MCP tool {namespaced_name}: {str(e)}"
                logger.error(error_msg)
                # Return an error ToolResult
                from llmproc.tools.tool_result import ToolResult

                return ToolResult.from_error(error_msg)

        # Register with the registry
        tool_registry.register_tool(namespaced_name, mcp_tool_handler, tool_def)

        # Mark as registered
        registered_tools.add(namespaced_name)


def format_tool_for_anthropic(
    tool: MCPTool, server_name: str | None = None
) -> dict[str, Any]:
    """Format a tool for Anthropic API.

    Args:
        tool: Tool object from MCP registry
        server_name: Optional server name for proper namespacing

    Returns:
        Dictionary with tool information formatted for Anthropic API
    """
    # Create namespaced name with server prefix
    namespaced_name = f"{server_name}__{tool.name}" if server_name else tool.name

    # Ensure input schema has required fields
    input_schema = tool.inputSchema.copy() if tool.inputSchema else {}
    if "type" not in input_schema:
        input_schema["type"] = "object"
    if "properties" not in input_schema:
        input_schema["properties"] = {}

    # Create the tool definition
    return {
        "name": namespaced_name,
        "description": tool.description,
        "input_schema": input_schema,
    }
