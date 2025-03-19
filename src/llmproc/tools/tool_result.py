"""ToolResult class for standardizing tool execution results."""

import json
from typing import Any, Dict, List, Optional, Union


class ToolResult:
    """A standardized result from tool execution.

    This class provides a consistent format for tool results across different types
    of tools (MCP tools, system tools like spawn/fork, etc.). It matches both the
    format expected by the Anthropic API for tool results and what is returned by
    MCP servers.

    Attributes:
        content: The result content from the tool execution
        is_error: Boolean flag indicating if the tool execution resulted in an error
    """

    def __init__(
        self,
        content: str | dict[str, Any] | list[dict[str, Any]] | None = None,
        is_error: bool = False,
    ):
        """Initialize a ToolResult.

        Args:
            content: The result content from the tool execution
            is_error: Boolean flag indicating if the tool execution resulted in an error
        """
        self.content = content
        self.is_error = is_error

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for the Anthropic API.

        Returns:
            Dictionary representation with content and is_error fields
        """
        # Convert content to appropriate string format
        content_value = self.content

        # Handle None case
        if content_value is None:
            content_value = ""
        # Handle dictionary and list by JSON serializing
        elif isinstance(content_value, dict | list):
            try:
                content_value = json.dumps(content_value)
            except (TypeError, ValueError):
                # If JSON serialization fails, use string representation
                content_value = str(content_value)
        # Handle other non-string objects
        elif not isinstance(content_value, str):
            content_value = str(content_value)

        result = {"content": content_value, "is_error": self.is_error}
        return result

    @classmethod
    def from_error(cls, error_message: str) -> "ToolResult":
        """Create a ToolResult instance from an error message.

        Args:
            error_message: The error message to include in the content

        Returns:
            A ToolResult instance marked as an error
        """
        return cls(content=error_message, is_error=True)

    @classmethod
    def from_success(cls, content: Any) -> "ToolResult":
        """Create a ToolResult instance from successful content.

        Args:
            content: The content to include in the result

        Returns:
            A ToolResult instance marked as successful
        """
        return cls(content=content, is_error=False)

    def __str__(self) -> str:
        """String representation of ToolResult.

        Returns:
            A string representation of the result
        """
        return f"ToolResult(content={self.content}, is_error={self.is_error})"
