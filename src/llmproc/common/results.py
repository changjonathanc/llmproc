"""Common result types for llmproc.

This module contains fundamental result types used throughout the library.
These classes should have minimal dependencies to avoid circular imports.
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Any


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


@dataclass
class RunResult:
    """Contains metadata about a process run.

    This class captures information about an LLMProcess run, including:
    - API call information (raw responses from API providers)
    - Tool call information
    - Timing information for the run

    Note About Tool Tracking:
    This class maintains two synchronized collections for tool calls:
    - tool_calls: List of (name, args, result) tuples - simple format for basic tracking
    - tool_call_infos: List of detailed dictionaries with comprehensive tool call information

    The add_tool_call() method populates both collections to ensure they remain in sync.
    """

    # Basic attributes
    process: Any = None
    iterations: int = 0
    tool_calls: list[tuple[str, dict, Any]] = field(default_factory=list)
    last_message: str = ""
    token_counts: dict[str, int] = field(default_factory=dict)

    # Primary data storage
    api_call_infos: list[dict[str, Any]] = field(default_factory=list)
    tool_call_infos: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: int = 0

    @property
    def api_calls(self) -> int:
        """Get number of API calls made."""
        return len(self.api_call_infos)

    @property
    def total_interactions(self) -> int:
        """Get total number of interactions (API calls + tool calls)."""
        return self.api_calls + len(self.tool_calls)

    def add_api_call(self, info: dict[str, Any]) -> None:
        """Record information about an API call."""
        self.api_call_infos.append(info)

    def add_tool_call(self, info: dict[str, Any]) -> None:
        """Record information about a tool call."""
        # Store the tool call info in the dedicated collection
        self.tool_call_infos.append(info)

        # Also update the tool_calls list to maintain consistency
        # This is not for backward compatibility but for ensuring both collections
        # track the same information
        if "tool_name" in info:
            # Add a tuple in the format (name, args, result) expected by tool_calls
            args = info.get("args", {})
            self.tool_calls.append((info["tool_name"], args, None))

    def complete(self) -> "RunResult":
        """Mark the run as complete and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)
        return self

    @property
    def cached_tokens(self) -> int:
        """Return the total number of tokens retrieved from cache."""
        total = 0
        for call in self.api_call_infos:
            usage = call.get("usage", {})
            # Handle both dictionary and object access
            if hasattr(usage, "cache_read_input_tokens"):
                total += getattr(usage, "cache_read_input_tokens", 0)
            elif isinstance(usage, dict):
                total += usage.get("cache_read_input_tokens", 0)
        return total

    @property
    def cache_write_tokens(self) -> int:
        """Return the total number of tokens written to cache."""
        total = 0
        for call in self.api_call_infos:
            usage = call.get("usage", {})
            # Handle both dictionary and object access
            if hasattr(usage, "cache_creation_input_tokens"):
                total += getattr(usage, "cache_creation_input_tokens", 0)
            elif isinstance(usage, dict):
                total += usage.get("cache_creation_input_tokens", 0)
        return total

    @property
    def cache_savings(self) -> float:
        """
        Return the estimated cost savings from cache usage.

        Cached tokens cost only 10% of regular input tokens,
        so savings is calculated as 90% of the cached token cost.
        """
        if not hasattr(self, "cached_tokens") or not self.cached_tokens:
            return 0.0

        # Cached tokens are 90% cheaper than regular input tokens
        return 0.9 * self.cached_tokens

    @property
    def input_tokens(self) -> int:
        """Return the total number of input tokens used."""
        total = 0
        for call in self.api_call_infos:
            usage = call.get("usage", {})
            # Handle both dictionary and object access
            if hasattr(usage, "input_tokens"):
                total += getattr(usage, "input_tokens", 0)
            elif isinstance(usage, dict):
                total += usage.get("input_tokens", 0)
        return total

    @property
    def output_tokens(self) -> int:
        """Return the total number of output tokens used."""
        total = 0
        for call in self.api_call_infos:
            usage = call.get("usage", {})
            # Handle both dictionary and object access
            if hasattr(usage, "output_tokens"):
                total += getattr(usage, "output_tokens", 0)
            elif isinstance(usage, dict):
                total += usage.get("output_tokens", 0)
        return total

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens used."""
        return self.input_tokens + self.output_tokens

    def __repr__(self) -> str:
        """Create a string representation of the run result."""
        status = "complete" if self.end_time else "in progress"
        duration = f"{self.duration_ms}ms" if self.end_time else "ongoing"
        cache_stats = ""
        token_stats = ""

        if self.cached_tokens > 0 or self.cache_write_tokens > 0:
            cache_stats = f", cached_tokens={self.cached_tokens}, cache_write_tokens={self.cache_write_tokens}"

        if self.total_tokens > 0:
            token_stats = f", input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, total_tokens={self.total_tokens}"

        return f"RunResult({status}, api_calls={self.api_calls}, tool_calls={len(self.tool_calls)}, total={self.total_interactions}{cache_stats}{token_stats}, duration={duration})"
