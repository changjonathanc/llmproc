from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from llmproc.common.results import ToolResult
from llmproc.plugin.plugin_runner import PluginRunner
from llmproc.plugin.plugin_utils import (
    call_plugin,
    filter_callback_parameters,
    has_plugin_method,
)

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from llmproc.plugin.datatypes import ResponseHookResult, ToolCallHookResult
from llmproc.plugin.events import (
    EVENT_CATEGORIES,
    CallbackEvent,
    EventCategory,
    HookEvent,
)

logger = logging.getLogger(__name__)


class PluginEventRunner(PluginRunner):
    """Unified runner for both hooks and callbacks."""

    def __init__(self, submit, plugins: Iterable[Any] | None = None) -> None:
        super().__init__(submit, plugins)
        self.plugins = self._plugins

    def __iter__(self):
        """Iterate over registered plugins."""
        return iter(self._plugins)

    async def run_event(self, event: str | HookEvent, *args: Any, **kwargs: Any) -> Any:
        """Run an event across all plugins and return the aggregated result."""
        key = event
        name = event
        if isinstance(event, HookEvent):
            key = event
            name = event.value
        category = EVENT_CATEGORIES.get(key)
        if category is EventCategory.OBSERVATIONAL:
            await self._trigger_callback(name, *args, **kwargs)
            return None
        raise RuntimeError(
            "run_event should only be used for observational events; "
            "call the dedicated hook method for behavioral events"
        )

    # ------------------------------------------------------------------
    # Observational callbacks
    # ------------------------------------------------------------------
    async def _trigger_callback(self, event: str, process, **kwargs: Any) -> None:
        """Invoke a callback event for all plugins.

        Args:
            event: Name of the callback to run.
            process: The active process instance providing context.
            **kwargs: Extra parameters forwarded to the callback.

        Asynchronous callback results are awaited synchronously.
        """
        if not self._plugins:
            return

        event_kwargs = kwargs.copy()
        event_kwargs["process"] = process

        for plugin in self._plugins:
            if not has_plugin_method(plugin, event):
                continue

            method = getattr(plugin, event)
            filtered = filter_callback_parameters(method, event_kwargs)
            try:
                result = method(**filtered)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error in %s callback: %s", event, exc)

    # ------------------------------------------------------------------
    # Behavioral hook methods
    # ------------------------------------------------------------------
    async def user_input(self, user_input: str, process) -> str:
        current = user_input
        for plugin in self._plugins:
            result = await self._call_async(plugin, HookEvent.USER_INPUT.value, current, process, propagate=True)
            if result is not None:
                current = result
        return current

    async def tool_call(self, process, tool_name: str, args: dict) -> tuple[str, dict, bool, ToolResult | None]:
        from llmproc.plugin.datatypes import ToolCallHookResult

        current_args = args
        for plugin in self._plugins:
            result = await self._call_async(
                plugin,
                HookEvent.TOOL_CALL.value,
                tool_name,
                current_args,
                process,
                propagate=True,
            )
            if isinstance(result, ToolCallHookResult):
                if result.modified_args is not None:
                    current_args = result.modified_args
                if result.skip_execution:
                    return tool_name, current_args, True, result.skip_result
        return tool_name, current_args, False, None

    async def tool_result(self, result: ToolResult, process, tool_name: str) -> ToolResult:
        current = result
        for plugin in self._plugins:
            modified = await self._call_async(
                plugin,
                HookEvent.TOOL_RESULT.value,
                tool_name,
                current,
                process,
                propagate=True,
            )
            if modified is not None:
                current = modified
        return current

    async def system_prompt(self, prompt: str, process) -> str:
        current = prompt
        for plugin in self._plugins:
            modified = await self._call_async(
                plugin,
                HookEvent.SYSTEM_PROMPT.value,
                current,
                process,
                propagate=True,
            )
            if modified is not None:
                current = modified
        return current

    async def response(self, process, content: str) -> ResponseHookResult | None:
        from llmproc.plugin.datatypes import ResponseHookResult

        result = None
        for plugin in self._plugins:
            hook_result = await self._call_async(
                plugin,
                HookEvent.RESPONSE.value,
                content,
                process,
                propagate=True,
            )
            if isinstance(hook_result, ResponseHookResult) or hook_result is not None:
                result = hook_result
                if getattr(hook_result, "stop", False):
                    break
        return result

    def provide_tools(self) -> list[Callable]:
        tools: list[Callable] = []
        for plugin in self._plugins:
            if has_plugin_method(plugin, HookEvent.PROVIDE_TOOLS.value):
                provided = call_plugin(plugin, HookEvent.PROVIDE_TOOLS.value) or []
                tools.extend(provided)
        return tools


__all__ = ["PluginEventRunner"]
