"""Unified protocol for callbacks and hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from llmproc.common.results import RunResult, ToolResult
    from llmproc.plugin.datatypes import ResponseHookResult, ToolCallHookResult


@runtime_checkable
class PluginProtocol(Protocol):
    """Complete interface for plugins.

    Plugins may implement any subset of these methods. All methods are optional
    and missing methods are simply ignored at runtime. This protocol is intended
    purely for IDE support and static type checking; inheritance is optional.
    """

    # ------------------------------------------------------------------
    # Observational callbacks
    # ------------------------------------------------------------------
    def tool_start(self, tool_name: str, tool_args: dict, *, process) -> None:
        """Called when a tool execution starts."""
        ...

    def api_stream_block(self, block, *, process) -> None:
        """Called for each raw streaming block before handling."""
        ...

    def tool_end(self, tool_name: str, result: ToolResult, *, process) -> None:
        """Called when a tool finishes executing."""
        ...

    def response(self, content: str, *, process) -> None:
        """Called when the model generates a response."""
        ...

    def api_request(self, api_request: dict, *, process) -> None:
        """Called before sending a request to the provider."""
        ...

    def api_response(self, response: dict, *, process) -> None:
        """Called after receiving a response from the provider."""
        ...

    def turn_start(self, *, process, run_result: Optional[RunResult] = None) -> None:
        """Called at the start of each conversation turn."""
        ...

    def turn_end(self, response: str, tool_results: list[ToolResult], *, process) -> None:
        """Called at the end of a conversation turn."""
        ...

    def run_end(self, run_result: RunResult, *, process) -> None:
        """Called when ``LLMProcess.run`` completes."""
        ...

    # ------------------------------------------------------------------
    # Behavioral hook methods
    # ------------------------------------------------------------------
    def hook_user_input(self, user_input: str, process) -> str | None:
        """Modify user input before processing."""
        ...

    def hook_tool_call(self, tool_name: str, args: dict, process) -> Optional[ToolCallHookResult]:
        """Modify or block a tool call before execution."""
        ...

    def hook_tool_result(self, tool_name: str, result: ToolResult, process) -> Optional[ToolResult]:
        """Modify tool results after execution."""
        ...

    def hook_system_prompt(self, system_prompt: str, process) -> str | None:
        """Modify the system prompt before sending it to the model."""
        ...

    def hook_response(self, response: str, process) -> Optional[ResponseHookResult]:
        """Observe model responses and optionally stop generation."""
        ...

    def hook_provide_tools(self) -> list:
        """Return additional tools to register for this program."""
        ...


__all__ = ["PluginProtocol"]
