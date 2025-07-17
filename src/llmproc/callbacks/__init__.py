"""Flexible callback management package.

This package provides LLMProc's flexible callback system with Flask/pytest-style
parameter injection. Callbacks only need to declare the parameters they actually use!

Key Features:
- **Flexible signatures**: Declare only the parameters you need
- **Performance benefits**: No unnecessary parameter processing with caching
- **Backward compatibility**: Legacy ``*, process`` signatures still work
- **Mix and match**: Use different signature styles freely

Callbacks are registered with ``process.add_plugins(obj)`` which also handles
hooks implemented on the same object.

Example usage:
    class FlexibleCallbacks:
        # Basic pattern: minimal signature
        def tool_start(self, tool_name):
            print(f"Tool starting: {tool_name}")

        # Selective pattern: choose what you need
        def tool_end(self, tool_name, result):
            print(f"Tool completed: {tool_name}")

        # Full context pattern: when you need process access
        def response(self, content, process):
            tokens = process.count_tokens()
            print(f"Response: {content[:50]}... ({tokens} tokens)")

        # Legacy pattern: still works for compatibility
        def api_request(self, api_request, *, process):
            print(f"API request to {process.model_name}")

    process.add_plugins(FlexibleCallbacks())

Available parameters by event:
- tool_start: tool_name, tool_args, process
- tool_end: tool_name, result, process
- response: content, process
- turn_start: process, run_result (optional)
- turn_end: response, tool_results, process
- api_request: api_request, process
- api_response: response, process
- run_end: run_result, process
"""

from llmproc.plugin.events import CallbackEvent
from llmproc.plugin.plugin_utils import filter_callback_parameters

__all__ = [
    "CallbackEvent",
    "filter_callback_parameters",
]
