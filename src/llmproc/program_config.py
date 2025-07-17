"""Configuration mixin for :class:`LLMProgram`.

This module contains helper methods that mutate a program's
configuration without creating a process. Splitting these methods
into a mixin clarifies which methods only modify program settings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from llmproc.common.access_control import AccessLevel  # noqa: F401 - used for docs
from llmproc.config.tool import ToolConfig
from llmproc.plugin.events import CallbackEvent
from llmproc.plugin.plugin_utils import call_plugin
from llmproc.plugins.override_utils import apply_tool_overrides
from llmproc.tools.mcp import MCPServerTools
from llmproc.tools.utils import convert_to_callables

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from llmproc.program import LLMProgram


class ProgramConfigMixin:
    """Mixin providing configuration helper methods."""

    plugins: list[Any]

    def add_linked_program(self, name: str, program: LLMProgram, description: str = "") -> LLMProgram:
        """Link another program to this one."""
        from llmproc.plugins.spawn import SpawnPlugin

        plugin = None
        for p in self.plugins:
            if isinstance(p, SpawnPlugin):
                plugin = p
                break

        if plugin is None:
            plugin = SpawnPlugin()
            self.plugins.append(plugin)

        plugin.linked_programs[name] = program
        plugin.linked_program_descriptions[name] = description
        return self

    def configure_thinking(self, enabled: bool = True, budget_tokens: int = 4096) -> LLMProgram:
        """Configure Claude 3.7 thinking capability."""
        if self.parameters is None:
            self.parameters = {}
        self.parameters["thinking"] = {
            "type": "enabled" if enabled else "disabled",
            "budget_tokens": budget_tokens,
        }
        return self

    def enable_token_efficient_tools(self) -> LLMProgram:
        """Enable token-efficient tool use for Claude 3.7 models."""
        if self.parameters is None:
            self.parameters = {}
        if "extra_headers" not in self.parameters:
            self.parameters["extra_headers"] = {}
        self.parameters["extra_headers"]["anthropic-beta"] = "token-efficient-tools-2025-02-19"
        return self

    def add_plugins(self, *plugins: Any) -> LLMProgram:
        """Register plugins for this program.

        Plugins can be:
        - Callable functions/objects
        - Objects implementing hook_* methods (behavioral)
        - Objects implementing callback methods (observational)
        """
        callback_methods = {event.value for event in CallbackEvent}

        for plugin in plugins:
            if not callable(plugin):
                plugin_methods = set(dir(plugin))
                has_hook = any(attr.startswith("hook_") for attr in plugin_methods)
                has_callback = bool(plugin_methods & callback_methods)

                if not has_hook and not has_callback:
                    raise ValueError(
                        f"Plugin {plugin} must be callable or implement a hook_* method "
                        f"or callback method ({', '.join(sorted(callback_methods))})"
                    )
            self.plugins.append(plugin)
        return self

    def register_tools(self, tools: list[str | Callable | MCPServerTools]) -> LLMProgram:
        """Register tools for use in the program."""
        if not isinstance(tools, list):
            tools = [tools]

        # Gather additional tools from plugins
        for plugin in self.plugins:
            provided = call_plugin(plugin, "hook_provide_tools") or []
            overrides = None
            if hasattr(plugin, "config") and hasattr(plugin.config, "tools"):
                overrides = plugin.config.tools
            provided = apply_tool_overrides(list(provided), overrides)
            tools.extend(provided)

        mcp_tools: list[MCPServerTools] = []
        other_tools: list[str | Callable | ToolConfig] = []

        for tool in tools:
            if isinstance(tool, MCPServerTools):
                mcp_tools.append(tool)
            else:
                other_tools.append(tool)

        if other_tools:
            callables = convert_to_callables(other_tools)
            if self.tools is None or any(not callable(t) and not isinstance(t, MCPServerTools) for t in self.tools):
                self.tools = []
            self.tools.extend(callables)

        if mcp_tools:
            if self.tools is None or any(not callable(t) and not isinstance(t, MCPServerTools) for t in self.tools):
                self.tools = [] if self.tools is None else [t for t in self.tools if callable(t)]
            self.tools.extend(mcp_tools)

        return self

    def get_registered_tools(self) -> list[str]:
        """Return the names of registered tools."""
        names: list[str] = []
        for item in self.tools or []:
            if isinstance(item, MCPServerTools):
                if item.tools == "all":
                    names.append(f"{item.server}:all")
                else:
                    for t in item.tools:
                        if isinstance(t, str):
                            names.append(t)
                        else:
                            names.append(t.name)
            elif callable(item):
                names.append(getattr(item, "__name__", str(item)))
        return names

    def set_user_prompt(self, prompt: str) -> LLMProgram:
        """Set a user prompt to be executed automatically when the program starts."""
        self.user_prompt = prompt
        return self

    def set_max_iterations(self, max_iterations: int) -> LLMProgram:
        """Set the default maximum number of iterations for this program."""
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        self.max_iterations = max_iterations
        return self

    def configure_mcp(
        self,
        config_path: str | None = None,
        servers: dict[str, dict] | None = None,
    ) -> LLMProgram:
        """Configure Model Context Protocol (MCP) server connection."""
        if config_path is not None:
            self.mcp_config_path = config_path
        if servers is not None:
            self.mcp_servers = servers
        return self
