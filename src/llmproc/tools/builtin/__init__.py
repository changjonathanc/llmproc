"""Builtin tool implementations for LLMProc.

This package contains the built-in tool implementations that are available
by default in LLMProc. These tools provide core functionality like file
operations, calculations, and process control.
"""

# Import all tools for re-export
from llmproc.tools.builtin.calculator import calculator
from llmproc.tools.builtin.fd_tools import fd_to_file_tool, read_fd_tool
from llmproc.tools.builtin.fork import fork_tool
from llmproc.tools.builtin.goto import handle_goto
from llmproc.tools.builtin.list_dir import list_dir
from llmproc.tools.builtin.read_file import read_file
from llmproc.tools.builtin.spawn import spawn_tool
from llmproc.tools.builtin.write_stderr import write_stderr_tool

# Central mapping of tool names to their implementations
# This provides a single source of truth for all builtin tools
BUILTIN_TOOLS = {
    "calculator": calculator,
    "read_file": read_file,
    "list_dir": list_dir,
    "fork": fork_tool,
    "goto": handle_goto,
    "spawn": spawn_tool,
    "read_fd": read_fd_tool,
    "fd_to_file": fd_to_file_tool,
    "write_stderr": write_stderr_tool,
}

__all__ = [
    "calculator",
    "fd_to_file_tool",
    "read_fd_tool",
    "fork_tool",
    "handle_goto",
    "list_dir",
    "read_file",
    "spawn_tool",
    "write_stderr_tool",
    "BUILTIN_TOOLS",  # Export the mapping
]
