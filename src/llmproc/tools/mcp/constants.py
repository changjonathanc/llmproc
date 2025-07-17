"""Constants for the MCP module.

This module defines constants used by the MCP manager and related tools.
"""

# Tool naming constants
MCP_TOOL_SEPARATOR = "__"

# Default timeout values
MCP_DEFAULT_TOOL_FETCH_TIMEOUT = 30.0
MCP_DEFAULT_TOOL_CALL_TIMEOUT = 30.0

# Log message constants
MCP_LOG_RETRY_FETCH = "Timeout fetching tools from MCP server '{server}' (attempt {attempt} of {max_attempts})"

# Error message constants
MCP_ERROR_INIT_FAILED = "Failed to initialize MCP tools: {error}"
MCP_ERROR_TOOL_FETCH_TIMEOUT = "Timeout fetching tools from MCP server '{server}' after {timeout:.1f} seconds. This typically happens when the server is slow to respond or not running properly. If you're using npx to run MCP servers, check if the package exists and is accessible. Consider increasing LLMPROC_TOOL_FETCH_TIMEOUT environment variable (current: {timeout:.1f}s) or check the server's status."
MCP_ERROR_TOOL_CALL_TIMEOUT = "Timeout calling tool '{tool}' on server '{server}' after {timeout:.1f} seconds. Consider checking server connectivity or increasing timeout."
