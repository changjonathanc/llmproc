"""Exceptions for the tools module.

This module defines custom exceptions for the tools module,
allowing for more specific error handling.
"""


class ToolError(Exception):
    """Base class for tool-related exceptions."""

    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found."""

    pass


class ToolRegistrationError(ToolError):
    """Raised when there's an error registering a tool."""

    pass


class ToolExecutionError(ToolError):
    """Raised when a tool execution fails."""

    pass


class ToolConfigurationError(ToolError):
    """Raised when a tool configuration is invalid."""

    pass
