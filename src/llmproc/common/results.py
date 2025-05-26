"""Common result types for llmproc.

This module contains fundamental result types used throughout the library.
These classes should have minimal dependencies to avoid circular imports.
"""

import json
import time
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
        abort_execution: Boolean flag indicating if the executor should stop processing further tools
    """

    def __init__(
        self,
        content: str | dict[str, Any] | list[dict[str, Any]] | None = None,
        is_error: bool = False,
        abort_execution: bool = False,
    ):
        """Initialize a ToolResult.

        Args:
            content: The result content from the tool execution
            is_error: Boolean flag indicating if the tool execution resulted in an error
            abort_execution: If True, signals the executor to stop processing further tools
        """
        self.content = content
        self.is_error = is_error
        self.abort_execution = abort_execution

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

    @classmethod
    def from_abort(cls, content: Any) -> "ToolResult":
        """Create a ToolResult instance that signals execution should abort.

        Use this for tools like GOTO that require stopping further tool processing
        after they execute.

        Args:
            content: The content to include in the result

        Returns:
            A ToolResult instance with abort_execution=True
        """
        return cls(content=content, is_error=False, abort_execution=True)

    def __str__(self) -> str:
        """String representation of ToolResult.

        Returns:
            A string representation of the result
        """
        return f"ToolResult(content={self.content}, is_error={self.is_error}, abort_execution={self.abort_execution})"


@dataclass
class RunResult:
    """Contains metadata about a process run.

    This class captures information about an LLMProcess run, including:
    - API call information (raw responses from API providers)
    - Tool call information
    - Timing information for the run
    - Token usage statistics

    A fluent API is provided for building and manipulating run results.
    """

    # Basic attributes
    process: Any = None
    last_message: str = ""
    token_counts: dict[str, int] = field(default_factory=dict)

    # Primary data storage - simplified to just two collections
    api_call_infos: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # Timing information
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: int = 0

    # Token statistics - stored as private counters for performance
    _input_tokens: int = 0
    _output_tokens: int = 0
    _cached_tokens: int = 0
    _cache_write_tokens: int = 0

    @property
    def api_calls(self) -> int:
        """Get number of API calls made."""
        return len(self.api_call_infos)

    @property
    def total_interactions(self) -> int:
        """Get total number of interactions (API calls + tool calls)."""
        return self.api_calls + len(self.tool_calls)

    def add_api_call(self, info: dict[str, Any]) -> "RunResult":
        """Record information about an API call.

        Args:
            info: Dictionary with API call information

        Returns:
            self for method chaining
        """
        self.api_call_infos.append(info)

        # Update token statistics from usage info
        usage = info.get("usage", {})

        # Handle both dictionary and object access for token counts
        if hasattr(usage, "input_tokens"):
            self._input_tokens += getattr(usage, "input_tokens", 0)
        elif isinstance(usage, dict):
            self._input_tokens += usage.get("input_tokens", 0)

        if hasattr(usage, "output_tokens"):
            self._output_tokens += getattr(usage, "output_tokens", 0)
        elif isinstance(usage, dict):
            self._output_tokens += usage.get("output_tokens", 0)

        if hasattr(usage, "cache_read_input_tokens"):
            self._cached_tokens += getattr(usage, "cache_read_input_tokens", 0)
        elif isinstance(usage, dict):
            self._cached_tokens += usage.get("cache_read_input_tokens", 0)

        if hasattr(usage, "cache_creation_input_tokens"):
            self._cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0)
        elif isinstance(usage, dict):
            self._cache_write_tokens += usage.get("cache_creation_input_tokens", 0)

        return self

    def add_tool_call(self, name: str, args: dict = None) -> "RunResult":
        """Record a tool call.

        Args:
            name: The name of the tool
            args: The arguments passed to the tool

        Returns:
            self for method chaining
        """
        self.tool_calls.append(
            {
                "tool_name": name,
                "args": args or {},
            }
        )
        return self

    def set_last_message(self, text: str) -> "RunResult":
        """Set the last message from the assistant.

        Args:
            text: The text of the last message

        Returns:
            self for method chaining
        """
        self.last_message = text
        return self

    def complete(self) -> "RunResult":
        """Mark the run as complete and calculate duration.

        Returns:
            self for method chaining
        """
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)
        return self

    def finish(self) -> "RunResult":
        """Alias for complete() for API consistency.

        Returns:
            self for method chaining
        """
        return self.complete()

    @property
    def cached_tokens(self) -> int:
        """Return the total number of tokens retrieved from cache."""
        return self._cached_tokens

    @property
    def cache_write_tokens(self) -> int:
        """Return the total number of tokens written to cache."""
        return self._cache_write_tokens

    @property
    def cache_savings(self) -> float:
        """
        Return the estimated cost savings from cache usage.

        Cached tokens cost only 10% of regular input tokens,
        so savings is calculated as 90% of the cached token cost.
        """
        if not self._cached_tokens:
            return 0.0

        # Cached tokens are 90% cheaper than regular input tokens
        return 0.9 * self._cached_tokens

    @property
    def input_tokens(self) -> int:
        """Return the total number of input tokens used."""
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        """Return the total number of output tokens used."""
        return self._output_tokens

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens used."""
        return self._input_tokens + self._output_tokens

    def __repr__(self) -> str:
        """Create a string representation of the run result."""
        status = "complete" if self.end_time else "in progress"
        duration = f"{self.duration_ms}ms" if self.end_time else "ongoing"
        cache_stats = ""
        token_stats = ""

        if self._cached_tokens > 0 or self._cache_write_tokens > 0:
            cache_stats = f", cached_tokens={self._cached_tokens}, cache_write_tokens={self._cache_write_tokens}"

        if self._input_tokens + self._output_tokens > 0:
            token_stats = f", input_tokens={self._input_tokens}, output_tokens={self._output_tokens}, total_tokens={self._input_tokens + self._output_tokens}"

        return f"RunResult({status}, api_calls={self.api_calls}, tool_calls={len(self.tool_calls)}, total={self.total_interactions}{cache_stats}{token_stats}, duration={duration})"
