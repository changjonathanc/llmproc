from dataclasses import dataclass

from llmproc.common.results import ToolResult


@dataclass
class ToolCallHookResult:
    """Result from a tool call hook."""

    modified_args: dict | None = None
    skip_execution: bool = False
    skip_result: ToolResult | None = None


@dataclass
class ResponseHookResult:
    """Result from a response hook."""

    stop: bool = False
    commit_current: bool = True


__all__ = ["ToolCallHookResult", "ResponseHookResult"]
