"""Improved hook execution package.

This package provides behavioral modification capabilities for LLMProcess execution.
Hooks run via the unified :class:`PluginEventRunner` alongside observational callbacks.
They can modify data and execution flow while errors propagate normally for fail-fast behavior.

Hooks and callbacks are both registered with ``process.add_plugins()``.

Example hook usage:
    class MyPlugin:
        def hook_user_input(self, user_input: str, process) -> str | None:
            '''Add timestamp to user input.'''
            return f"[{datetime.now()}] {user_input}"

        def hook_tool_call(self, tool_name: str, args: dict, process) -> ToolCallHookResult | None:
            '''Block dangerous tools.'''
            if tool_name in ["rm", "delete"]:
                return ToolCallHookResult(skip_execution=True,
                                        skip_result=ToolResult.from_error("Tool blocked"))
            return None

        def hook_tool_result(self, tool_name: str, result: ToolResult, process) -> ToolResult | None:
            '''Truncate long results.'''
            if len(result.content) > 1000:
                return ToolResult(content=result.content[:1000] + "... [truncated]")
            return None

    process.add_plugins(MyPlugin())  # Same registration as callbacks
"""

from llmproc.plugin.datatypes import ResponseHookResult, ToolCallHookResult

__all__ = [
    "ToolCallHookResult",
    "ResponseHookResult",
]
