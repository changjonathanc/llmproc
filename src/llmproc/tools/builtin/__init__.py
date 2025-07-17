"""Builtin tool implementations for LLMProc.

This package contains the built-in tool implementations that are available
by default in LLMProc. These tools provide core functionality like file
operations, calculations, and process control.
"""

from collections.abc import Callable

from llmproc.plugins.spawn import spawn_tool

# Import all tools for re-export
from llmproc.tools.builtin.ast_grep import ast_grep
from llmproc.tools.builtin.calculator import calculator
from llmproc.tools.builtin.fork import fork_tool
from llmproc.tools.builtin.list_dir import list_dir
from llmproc.tools.builtin.read_file import read_file

# Central mapping of tool names to their implementations
# This provides a single source of truth for all builtin tools
BUILTIN_TOOLS: dict[str, Callable] = {
    "calculator": calculator,
    "read_file": read_file,
    "list_dir": list_dir,
    "fork": fork_tool,
    "spawn": spawn_tool,
    "ast_grep": ast_grep,
}


def add_builtin_tool(name: str, func: Callable) -> None:
    """Register an additional builtin tool.

    This allows external libraries to expose builtin tools that can be enabled
    via configuration just like the core tools shipped with LLMProc.
    """
    if not callable(func):
        raise TypeError("func must be callable")
    if name in BUILTIN_TOOLS:
        raise ValueError(f"Builtin tool '{name}' already exists")
    BUILTIN_TOOLS[name] = func


__all__ = [
    "calculator",
    "fork_tool",
    "list_dir",
    "read_file",
    "ast_grep",
    "spawn_tool",
    "BUILTIN_TOOLS",  # Export the mapping
    "add_builtin_tool",
]
