"""Result types for LLMProcess executions."""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    """Contains metadata about a process run.

    This class captures information about an LLMProcess run, including:
    - API call information (raw responses from API providers)
    - Tool call information
    - Timing information for the run

    Attributes:
        api_call_infos: List of raw API response data, including usage metrics
        tool_call_infos: List of tool call data, including tool names and args
        api_calls: Number of API calls made during the run
        start_time: When the run started (timestamp)
        end_time: When the run completed (timestamp)
        duration_ms: Duration of the run in milliseconds
    """

    api_call_infos: list[dict[str, Any]] = field(default_factory=list)
    tool_call_infos: list[dict[str, Any]] = field(default_factory=list)
    api_calls: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: int = 0
    
    @property
    def tool_calls(self) -> int:
        """Get the number of tool calls made during the run.
        
        Returns:
            Number of tool calls
        """
        return len(self.tool_call_infos)
    
    @property
    def total_interactions(self) -> int:
        """Get the total number of interactions (API calls + tool calls).
        
        Returns:
            Total number of interactions
        """
        return self.api_calls + self.tool_calls

    def add_api_call(self, info: dict[str, Any]) -> None:
        """Record information about an API call.

        Args:
            info: Raw API response information, such as usage metrics
        """
        self.api_call_infos.append(info)
        self.api_calls += 1
        
    def add_tool_call(self, info: dict[str, Any]) -> None:
        """Record information about a tool call.
        
        Args:
            info: Tool call information, including tool name and args
        """
        self.tool_call_infos.append(info)

    def complete(self) -> "RunResult":
        """Mark the run as complete and calculate duration.

        Returns:
            Self, for method chaining
        """
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)
        return self

    def __repr__(self) -> str:
        """Create a string representation of the run result."""
        status = "complete" if self.end_time else "in progress"
        duration = f"{self.duration_ms}ms" if self.end_time else "ongoing"
        return (f"RunResult({status}, api_calls={self.api_calls}, "
                f"tool_calls={self.tool_calls}, total={self.total_interactions}, "
                f"duration={duration})")
