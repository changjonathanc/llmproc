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
            
        return (f"RunResult({status}, api_calls={self.api_calls}, "
                f"tool_calls={self.tool_calls}, total={self.total_interactions}{cache_stats}{token_stats}, "
                f"duration={duration})")
