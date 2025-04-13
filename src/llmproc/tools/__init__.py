"""Tools for LLMProcess.

This module provides system tools that can be used by LLMProcess instances.
It also provides a registry to retrieve tool handlers and schemas by name.
"""

import logging

from llmproc.common.results import ToolResult

# Import file descriptor constants directly from constants module
from llmproc.file_descriptors.constants import (
    FD_TO_FILE_TOOL_DEF as fd_to_file_tool_def,
)
from llmproc.file_descriptors.constants import (
    FILE_DESCRIPTOR_INSTRUCTIONS as file_descriptor_instructions,
)
from llmproc.file_descriptors.constants import (
    READ_FD_TOOL_DEF as read_fd_tool_def,
)
from llmproc.file_descriptors.constants import (
    REFERENCE_INSTRUCTIONS as reference_instructions,
)
from llmproc.file_descriptors.constants import (
    USER_INPUT_INSTRUCTIONS as fd_user_input_instructions,
)
from llmproc.tools.builtin.calculator import calculator
from llmproc.tools.builtin.fd_tools import fd_to_file_tool, read_fd_tool
from llmproc.tools.builtin.fork import fork_tool, fork_tool_def

# Import from integration module (replaces old utils imports)
from llmproc.tools.builtin.integration import (
    copy_tool_from_source_to_target,
    register_fork_tool,
    register_spawn_tool,
    register_system_tools,
)
from llmproc.tools.builtin.integration import (
    register_fd_tool as register_file_descriptor_tools,
)
from llmproc.tools.builtin.list_dir import list_dir
from llmproc.tools.builtin.read_file import read_file
from llmproc.tools.builtin.spawn import SPAWN_TOOL_SCHEMA as spawn_tool_def
from llmproc.tools.builtin.spawn import spawn_tool

# Import file descriptor instructions
# The instruction text provides guidance on how to use file descriptors in prompts
# Import tools registry
# Import all tools - these imports will register themselves
from . import registry_data
from .exceptions import (
    ToolConfigurationError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
)
from .function_tools import create_tool_from_function, register_tool
from .tool_manager import ToolManager
from .tool_registry import ToolHandler, ToolRegistry, ToolSchema

# Register all function-based tools in the central registry
registry_data.register("calculator", calculator)
registry_data.register("read_file", read_file)
registry_data.register("list_dir", list_dir)
# Add new function-based tools here when needed

# Set up logger
logger = logging.getLogger(__name__)

# Define system tools dictionary
_SYSTEM_TOOLS = {
    "spawn": (spawn_tool, spawn_tool_def),
    "fork": (fork_tool, fork_tool_def),
    # Calculator and read_file now use function-based implementations
    # defined with @register_tool - function and schema are created at runtime
    "read_fd": (read_fd_tool, read_fd_tool_def),
    "fd_to_file": (fd_to_file_tool, fd_to_file_tool_def),
}

# Export all tools and utilities
__all__ = [
    # Function-based tools
    "calculator",
    "read_file",
    "list_dir",
    # Add new function-based tools to exports here
    # Special tools
    "spawn_tool",
    "spawn_tool_def",
    "fork_tool",
    "fork_tool_def",
    # File descriptor tools
    "read_fd_tool",
    "read_fd_tool_def",
    "fd_to_file_tool",
    "fd_to_file_tool_def",
    # Instructions
    "file_descriptor_instructions",
    "fd_user_input_instructions",
    "reference_instructions",
    # Utilities (from integration)
    "register_system_tools",
    "register_spawn_tool",
    "register_fork_tool",
    "register_file_descriptor_tools",
    "copy_tool_from_source_to_target",
    # Classes
    "ToolSchema",
    "ToolHandler",
    "ToolRegistry",
    "ToolManager",
    "ToolError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolExecutionError",
    "ToolConfigurationError",
    # Functions from function_tools
    "ToolResult",
    "register_tool",
    "create_tool_from_function",
]
