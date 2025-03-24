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
    """

    api_call_infos: list[dict[str, Any]] = field(default_factory=list)
    tool_call_infos: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: int = 0
    
    @property
    def api_calls(self) -> int:
        return len(self.api_call_infos)
    
    @property
    def tool_calls(self) -> int:
        return len(self.tool_call_infos)
    
    @property
    def total_interactions(self) -> int:
        return self.api_calls + self.tool_calls

    def add_api_call(self, info: dict[str, Any]) -> None:
        """Record information about an API call."""
        self.api_call_infos.append(info)
        
    def add_tool_call(self, info: dict[str, Any]) -> None:
        """Record information about a tool call."""
        self.tool_call_infos.append(info)

    def complete(self) -> "RunResult":
        """Mark the run as complete and calculate duration."""
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
