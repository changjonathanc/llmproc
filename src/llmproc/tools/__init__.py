"""Tools for LLMProcess.

This module provides system tools that can be used by LLMProcess instances.
It also provides a registry to retrieve tool handlers and schemas by name.
"""

import logging

from . import mcp
from .calculator import calculator
from .exceptions import (
    ToolError, 
    ToolNotFoundError, 
    ToolRegistrationError,
    ToolExecutionError,
    ToolConfigurationError
)
from .file_descriptor import (
    fd_to_file_tool, fd_to_file_tool_def,
    file_descriptor_instructions, file_descriptor_base_instructions,
    fd_user_input_instructions, reference_instructions,
    read_fd_tool, read_fd_tool_def
)
from .fork import fork_tool, fork_tool_def
from .function_tools import register_tool, create_tool_from_function
from .read_file import read_file
from .spawn import spawn_tool, spawn_tool_def
from .tool_manager import ToolManager
from .tool_registry import ToolRegistry, ToolSchema, ToolHandler
from .tool_result import ToolResult
from .utils import (
    get_tool,
    list_available_tools,
    register_system_tools,
    register_spawn_tool,
    register_fork_tool,
    register_calculator_tool,
    register_read_file_tool,
    register_file_descriptor_tools
)

# Set up logger
logger = logging.getLogger(__name__)


# Registry of available system tools (kept for backward compatibility)
_SYSTEM_TOOLS = {
    "spawn": (spawn_tool, spawn_tool_def),
    "fork": (fork_tool, fork_tool_def),
    # Calculator and read_file now use function-based implementations
    # defined with @register_tool - function and schema are created at runtime
    "read_fd": (read_fd_tool, read_fd_tool_def),
    "fd_to_file": (fd_to_file_tool, fd_to_file_tool_def),
}


__all__ = [
    "spawn_tool",
    "spawn_tool_def",
    "fork_tool",
    "fork_tool_def",
    "calculator",
    "read_fd_tool",
    "read_fd_tool_def",
    "fd_to_file_tool",
    "fd_to_file_tool_def",
    "read_file",
    "file_descriptor_instructions",
    "file_descriptor_base_instructions",
    "fd_user_input_instructions",
    "reference_instructions",
    "get_tool",
    "list_available_tools",
    "ToolSchema",
    "ToolHandler",
    "ToolRegistry",
    "ToolManager",
    "ToolError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolExecutionError",
    "ToolConfigurationError",
    "register_system_tools",
    "register_spawn_tool",
    "register_fork_tool",
    "register_calculator_tool",
    "register_read_file_tool",
    "register_file_descriptor_tools",
    "ToolResult",
    "mcp",
    "register_tool",
    "create_tool_from_function",
]