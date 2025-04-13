"""Builtin tool implementations for LLMProc.

This package contains the built-in tool implementations that are available
by default in LLMProc. These tools provide core functionality like file
operations, calculations, and process control.

Re-exporting tool functions for backward compatibility.
"""

# Import all tools for re-export
from llmproc.tools.builtin.calculator import calculator
from llmproc.tools.builtin.fd_tools import fd_to_file_tool, read_fd_tool
from llmproc.tools.builtin.fork import fork_tool, fork_tool_def
from llmproc.tools.builtin.goto import GOTO_TOOL_DEFINITION, handle_goto
from llmproc.tools.builtin.list_dir import list_dir
from llmproc.tools.builtin.read_file import read_file
from llmproc.tools.builtin.spawn import SPAWN_TOOL_SCHEMA, spawn_tool

__all__ = [
    "calculator",
    "fd_to_file_tool",
    "read_fd_tool",
    "fork_tool",
    "fork_tool_def",
    "handle_goto",
    "GOTO_TOOL_DEFINITION",
    "list_dir",
    "read_file",
    "spawn_tool",
    "SPAWN_TOOL_SCHEMA",
]
