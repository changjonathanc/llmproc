"""Configuration mixin for :class:`LLMProgram`.

This module contains helper methods that mutate a program's
configuration without creating a process. Splitting these methods
into a mixin clarifies which methods only modify program settings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llmproc.common.access_control import AccessLevel  # noqa: F401 - used for docs
from llmproc.common.metadata import attach_meta, get_tool_meta
from llmproc.config import EnvInfoConfig
from llmproc.config.tool import ToolConfig
from llmproc.tools import ToolManager
from llmproc.tools.builtin import BUILTIN_TOOLS
from llmproc.tools.mcp import MCPServerTools
from llmproc.tools.mcp.constants import MCP_TOOL_SEPARATOR


def convert_to_callables(tools: list[str | Callable | MCPServerTools | ToolConfig]) -> list[Callable]:
    """Return callable tools, ignoring ``MCPServerTools`` descriptors."""
    if not isinstance(tools, list):
        tools = [tools]

    result: list[Callable] = []
    for tool in tools:
        if isinstance(tool, str):
            if tool in BUILTIN_TOOLS:
                result.append(BUILTIN_TOOLS[tool])
            else:
                raise ValueError(f"Unknown tool name: '{tool}'")
        elif isinstance(tool, ToolConfig):
            name = tool.name
            if name in BUILTIN_TOOLS:
                func = BUILTIN_TOOLS[name]
                if tool.description is not None or tool.param_descriptions is not None:
                    meta = get_tool_meta(func)
                    if tool.description is not None:
                        meta.description = tool.description
                    if tool.param_descriptions is not None:
                        existing = dict(meta.param_descriptions or {})
                        existing.update(tool.param_descriptions)
                        meta.param_descriptions = existing
                    attach_meta(func, meta)
                result.append(func)
            else:
                raise ValueError(f"Unknown tool name: '{name}'")
        elif callable(tool):
            result.append(tool)
        elif isinstance(tool, MCPServerTools):
            pass
        else:
            raise ValueError(f"Expected string, callable, or MCPServerTools, got {type(tool)}")
    return result


class ProgramConfigMixin:
    """Mixin providing configuration helper methods."""

    tool_manager: ToolManager
    parameters: dict[str, Any] | None
    file_descriptor: dict[str, Any] | None
    mcp_config_path: str | None
    mcp_servers: dict[str, dict] | None
    env_info: EnvInfoConfig
    user_prompt: str | None
    max_iterations: int

    def add_linked_program(self, name: str, program: LLMProgram, description: str = "") -> LLMProgram:
        """Link another program to this one."""
        self.linked_programs[name] = program
        self.linked_program_descriptions[name] = description
        return self

    def add_preload_file(self, file_path: str) -> LLMProgram:
        """Add a file to preload into the system prompt."""
        self.preload_files.append(file_path)
        return self

    def configure_env_info(
        self, variables: list[str] | str = "all", env_vars: dict[str, str] | None = None
    ) -> LLMProgram:
        """Configure environment information sharing."""
        parsed = EnvInfoConfig(variables=variables)
        self.env_info.variables = parsed.variables
        if env_vars:
            self.env_info.env_vars.update(env_vars)
        return self

    def configure_file_descriptor(
        self,
        enabled: bool = True,
        max_direct_output_chars: int = 8000,
        default_page_size: int = 4000,
        max_input_chars: int = 8000,
        page_user_input: bool = True,
        enable_references: bool = True,
    ) -> LLMProgram:
        """Configure the file descriptor system."""
        self.file_descriptor = {
            "enabled": enabled,
            "max_direct_output_chars": max_direct_output_chars,
            "default_page_size": default_page_size,
            "max_input_chars": max_input_chars,
            "page_user_input": page_user_input,
            "enable_references": enable_references,
        }
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

    def register_tools(self, tools: list[str | Callable | MCPServerTools]) -> LLMProgram:
        """Register tools for use in the program."""
        if not isinstance(tools, list):
            tools = [tools]

        mcp_tools: list[MCPServerTools] = []
        other_tools: list[str | Callable | ToolConfig] = []
        alias_map: dict[str, str] = {}

        for tool in tools:
            if isinstance(tool, MCPServerTools):
                mcp_tools.append(tool)
                if tool.tools != "all" and isinstance(tool.tools, list):
                    for item in tool.tools:
                        if isinstance(item, ToolConfig) and item.alias:
                            alias_map[item.alias] = f"{tool.server}{MCP_TOOL_SEPARATOR}{item.name}"
            else:
                other_tools.append(tool)
                if isinstance(tool, ToolConfig) and tool.alias:
                    alias_map[tool.alias] = tool.name

        if other_tools:
            callables = convert_to_callables(other_tools)
            self.tool_manager.register_tools(callables)

        if mcp_tools:
            self.tool_manager.register_tools(mcp_tools)

        if alias_map:
            self.set_tool_aliases(alias_map)

        return self

    def get_registered_tools(self) -> list[str]:
        """Return the names of registered tools."""
        return self.tool_manager.get_registered_tools()

    def set_enabled_tools(self, tools: list[str | Callable]) -> LLMProgram:
        """Alias for register_tools for backward compatibility."""
        return self.register_tools(tools)

    def set_tool_aliases(self, aliases: dict[str, str]) -> LLMProgram:
        """Set tool aliases, merging with any existing aliases."""
        if not isinstance(aliases, dict):
            raise ValueError(f"Expected dictionary of aliases, got {type(aliases)}")

        targets: dict[str, str] = {}
        for alias, target in aliases.items():
            if target in targets:
                raise ValueError(
                    f"Multiple aliases point to the same target tool '{target}': '{targets[target]}'"
                    f" and '{alias}'. One-to-one mapping is required."
                )
            targets[target] = alias

        self.tool_manager.register_aliases(aliases)
        return self

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
