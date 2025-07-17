class MCPError(Exception):
    """Base class for MCP-related errors."""


class MCPConnectionsDisabledError(MCPError):
    """Raised when persistent connections are disabled."""


class MCPServerConnectionError(MCPError):
    """Raised when a server cannot be reached."""


class MCPToolsLoadingError(MCPError):
    """Raised when tools cannot be loaded from servers."""
