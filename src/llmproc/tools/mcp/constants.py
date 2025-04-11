"""Constants for the MCP module.

This module defines constants used by the MCP manager and related tools.
"""

# Tool naming constants
MCP_TOOL_SEPARATOR = "__"

# Log message constants
MCP_LOG_INITIALIZING_SERVERS = "Initializing {count} MCP servers: {servers}"
MCP_LOG_NO_SERVERS = "No MCP servers configured - skipping MCP initialization"
MCP_LOG_REGISTERED_SERVER_TOOLS = "Registered {count} tools from server '{server}'"
MCP_LOG_TOTAL_REGISTERED = "Registered a total of {count} MCP tools"
MCP_LOG_MCP_TOOL_NAMES = "MCP tool names: {names}"
MCP_LOG_ENABLED_TOOLS = "Enabled tools: {tools}"
MCP_LOG_NO_TOOLS_REGISTERED = "No MCP tools were registered despite having configuration"

# Error message constants
MCP_ERROR_INIT_FAILED = "Failed to initialize MCP tools: {error}"
MCP_ERROR_NO_TOOLS_REGISTERED = (
    "No MCP tools were registered despite having configuration. "
    "Check that the server names and tool names in your mcp_tools configuration exist. "
    "Servers config: {servers_config}"
)