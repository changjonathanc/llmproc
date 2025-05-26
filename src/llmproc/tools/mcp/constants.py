"""Constants for the MCP module.

This module defines constants used by the MCP manager and related tools.
"""

# Tool naming constants
MCP_TOOL_SEPARATOR = "__"

# Default timeout values
MCP_DEFAULT_TOOL_FETCH_TIMEOUT = 30.0
MCP_DEFAULT_TOOL_CALL_TIMEOUT = 30.0
MCP_MAX_FETCH_RETRIES = 2

# Log message constants
MCP_LOG_INITIALIZING_SERVERS = "Initializing {count} MCP servers: {servers}"
MCP_LOG_NO_SERVERS = "No MCP servers configured - skipping MCP initialization"
MCP_LOG_REGISTERED_SERVER_TOOLS = "Registered {count} tools from server '{server}'"
MCP_LOG_TOTAL_REGISTERED = "Registered a total of {count} MCP tools"
MCP_LOG_MCP_TOOL_NAMES = "MCP tool names: {names}"
MCP_LOG_ENABLED_TOOLS = "Enabled tools: {tools}"
MCP_LOG_NO_TOOLS_REGISTERED = "No MCP tools were registered despite having configuration"
MCP_LOG_RETRY_FETCH = "Timeout fetching tools from MCP server '{server}' (attempt {attempt} of {max_attempts})"

# Error message constants
MCP_ERROR_INIT_FAILED = "Failed to initialize MCP tools: {error}"
MCP_ERROR_NO_TOOLS_REGISTERED = "No MCP tools were registered despite having configuration. Check that the server names and tool names in your mcp_tools configuration exist. Servers config: {servers_config}"
MCP_ERROR_TOOL_FETCH_TIMEOUT = "Timeout fetching tools from MCP server '{server}' after {timeout:.1f} seconds. This typically happens when the server is slow to respond or not running properly. If you're using npx to run MCP servers, check if the package exists and is accessible. Consider increasing LLMPROC_TOOL_FETCH_TIMEOUT environment variable (current: {timeout:.1f}s) or check the server's status."
MCP_ERROR_TOOL_CALL_TIMEOUT = "Timeout calling tool '{tool}' on server '{server}' after {timeout:.1f} seconds. Consider checking server connectivity or increasing timeout."
