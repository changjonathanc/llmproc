"""Tools for LLMProcess.

This module provides system tools that can be used by LLMProcess instances.
It also provides a registry to retrieve tool handlers and schemas by name.
"""

import logging

# Core results dataclass
from llmproc.common.results import ToolResult

# Import file descriptor constants directly from constants module
from llmproc.plugins.file_descriptor.constants import (
    FILE_DESCRIPTOR_INSTRUCTIONS as file_descriptor_instructions,
)
from llmproc.plugins.file_descriptor.constants import (
    REFERENCE_INSTRUCTIONS as reference_instructions,
)
from llmproc.plugins.file_descriptor.constants import (
    USER_INPUT_INSTRUCTIONS as fd_user_input_instructions,
)
from llmproc.plugins.spawn import spawn_tool
from llmproc.tools.anthropic import WebSearchTool
from llmproc.tools.builtin import BUILTIN_TOOLS, add_builtin_tool
from llmproc.tools.builtin.calculator import calculator
from llmproc.tools.builtin.fork import fork_tool
from llmproc.tools.builtin.list_dir import list_dir
from llmproc.tools.builtin.read_file import read_file
from llmproc.tools.openai import OpenAIWebSearchTool

# Import file descriptor instructions
# The instruction text provides guidance on how to use file descriptors in prompts
# Import tools registry
# Import all tools - these imports will register themselves
from .core import Tool
from .function_schemas import create_schema_from_callable
from .function_tools import (
    create_handler_from_function,
    register_tool,
)
from .tool_manager import ToolManager
from .tool_registry import ToolHandler, ToolRegistry, ToolSchema

# Set up logger
logger = logging.getLogger(__name__)


# Export all tools and utilities
__all__ = [
    # Function-based tools
    "calculator",
    "read_file",
    "list_dir",
    # Add new function-based tools to exports here
    # Special tools
    "spawn_tool",
    "fork_tool",
    # Instructions
    "file_descriptor_instructions",
    "fd_user_input_instructions",
    "reference_instructions",
    # Classes
    "ToolSchema",
    "ToolHandler",
    "ToolRegistry",
    "Tool",
    "ToolManager",
    # Functions from function_tools
    "ToolResult",
    "register_tool",
    "create_handler_from_function",
    "create_schema_from_callable",
    "BUILTIN_TOOLS",
    "add_builtin_tool",
    "WebSearchTool",
    "OpenAIWebSearchTool",
]
