"""Integration functions for builtin tools.

This module provides functions for initializing and registering builtin tools,
enabling easier integration with the ToolManager system and following the
same pattern as the MCP integration module.
"""

import logging
from collections.abc import Callable
from typing import Any, Optional

from llmproc.common.results import ToolResult
from llmproc.file_descriptors.constants import (
    FD_RELATED_TOOLS,
    FD_TO_FILE_TOOL_DEF,
    READ_FD_TOOL_DEF,
)

# Import builtin tool components
from llmproc.tools.builtin.calculator import calculator
from llmproc.tools.builtin.fd_tools import fd_to_file_tool, read_fd_tool
from llmproc.tools.builtin.fork import fork_tool, fork_tool_def
from llmproc.tools.builtin.goto import GOTO_TOOL_DEFINITION, handle_goto
from llmproc.tools.builtin.list_dir import list_dir
from llmproc.tools.builtin.read_file import read_file
from llmproc.tools.builtin.spawn import SPAWN_TOOL_SCHEMA, spawn_tool
from llmproc.tools.context_aware import is_context_aware
from llmproc.tools.registry_data import get_function_tool_names
from llmproc.tools.registry_helpers import extract_tool_components
from llmproc.tools.tool_registry import ToolRegistry

# Set up logger
logger = logging.getLogger(__name__)


def load_builtin_tools(registry: ToolRegistry) -> bool:
    """Load all available builtin tools to the provided registry.

    This is typically done once during initialization and doesn't depend on which tools are enabled.

    Args:
        registry: The registry to load builtin tools into

    Returns:
        True if registration succeeded, False otherwise
    """
    logger.info("Loading builtin tools into registry")

    # 1. Load context-aware tools with their actual handlers

    # Fork tool
    fork_def = fork_tool_def.copy()

    # Real handler is injected by the process executor, but we register a context-aware handler
    # that can signal the need for context
    async def fork_handler(prompts, runtime_context=None):
        if not runtime_context or "process" not in runtime_context:
            return ToolResult.from_error("Fork tool requires a process context.")
        return ToolResult.from_error(
            "Direct calls to fork_tool are not supported. This should be handled by the process executor."
        )

    registry.register_tool("fork", fork_handler, fork_def)

    # Spawn tool
    spawn_def = SPAWN_TOOL_SCHEMA.copy()
    # Register the real context-aware handler directly
    registry.register_tool("spawn", spawn_tool, spawn_def)

    # Goto tool
    goto_def = GOTO_TOOL_DEFINITION.copy()
    # Register the real context-aware handler directly
    registry.register_tool("goto", handle_goto, goto_def)

    # File descriptor tools
    # read_fd tool
    read_fd_def = READ_FD_TOOL_DEF.copy()
    # Register the real context-aware handler directly
    registry.register_tool("read_fd", read_fd_tool, read_fd_def)

    # fd_to_file tool
    fd_to_file_def = FD_TO_FILE_TOOL_DEF.copy()
    # Register the real context-aware handler directly
    registry.register_tool("fd_to_file", fd_to_file_tool, fd_to_file_def)

    # 2. Function-based tools that don't need runtime context
    # These are defined as functions and we call them directly
    # Calculator tool
    from llmproc.tools.function_tools import create_tool_from_function

    calc_handler, calc_def = create_tool_from_function(calculator)
    registry.register_tool("calculator", calc_handler, calc_def)

    # Read file tool
    read_file_handler, read_file_def = create_tool_from_function(read_file)
    registry.register_tool("read_file", read_file_handler, read_file_def)

    # List directory tool
    list_dir_handler, list_dir_def = create_tool_from_function(list_dir)
    registry.register_tool("list_dir", list_dir_handler, list_dir_def)

    logger.info("Finished loading builtin tools into registry")
    return True


def register_system_tools(
    source_registry: ToolRegistry,
    target_registry: ToolRegistry,
    enabled_tools: list[str],
    config: dict[str, Any],
) -> int:
    """Register system tools based on configuration.

    This function handles the registration of builtin tools from the source registry
    to the target registry, using the provided configuration.

    Args:
        source_registry: The registry containing builtin tool definitions
        target_registry: The registry to register tools into
        enabled_tools: List of tool names to enable
        config: Dictionary containing tool dependencies including:
            - fd_manager: File descriptor manager instance or None
            - linked_programs: Dictionary of linked programs
            - linked_program_descriptions: Dictionary of program descriptions
            - has_linked_programs: Whether linked programs are available
            - provider: The LLM provider name

    Returns:
        int: Number of tools registered
    """
    logger.info(
        f"Starting system tools registration based on enabled list: {enabled_tools}"
    )

    # Extract configuration components
    fd_manager = config.get("fd_manager")
    linked_programs = config.get("linked_programs", {})
    linked_program_descriptions = config.get("linked_program_descriptions", {})
    has_linked_programs = config.get("has_linked_programs", False)
    fd_enabled = fd_manager is not None

    # Get function-based tools from registry
    function_tool_names = get_function_tool_names()

    # Dictionary mapping tool names to their registration functions
    tool_registrars = {
        # Basic tools
        "calculator": lambda name: copy_tool_from_source_to_target(
            source_registry, target_registry, name
        ),
        "read_file": lambda name: copy_tool_from_source_to_target(
            source_registry, target_registry, name
        ),
        "list_dir": lambda name: copy_tool_from_source_to_target(
            source_registry, target_registry, name
        ),
        # Special case tools
        "fork": lambda name: register_fork_tool(source_registry, target_registry, name),
        "goto": lambda name: register_goto_tool(source_registry, target_registry, name),
        "spawn": lambda name: register_spawn_tool(
            source_registry,
            target_registry,
            name,
            linked_programs,
            linked_program_descriptions,
        ),
        "read_fd": lambda name: register_fd_tool(
            source_registry, target_registry, name, fd_manager
        ),
        "fd_to_file": lambda name: register_fd_tool(
            source_registry, target_registry, name, fd_manager
        ),
    }

    # Add the read_fd tool to enabled_tools if fd_manager is available but not enabled explicitly
    # We only add read_fd by default as it's the basic functionality, fd_to_file requires explicit opt-in
    if fd_enabled and "read_fd" not in enabled_tools:
        logger.info(
            "File descriptor system is enabled, automatically enabling read_fd tool"
        )
        # Add to enabled tools list - it'll get registered in the loop below
        enabled_tools.append("read_fd")

    registered_count = 0
    # Register each enabled tool
    for tool_name in enabled_tools:
        if tool_name in tool_registrars:
            # Special cases for tools we haven't updated yet
            if tool_name == "spawn" and not has_linked_programs:
                logger.info(
                    "Skipping spawn tool registration: marked as enabled but no linked programs available"
                )
                continue

            # Call the appropriate registration function
            try:
                success = tool_registrars[tool_name](tool_name)
                if success:
                    registered_count += 1
                    logger.debug(f"Successfully registered system tool: {tool_name}")
                else:
                    logger.warning(f"Failed to register system tool: {tool_name}")
            except Exception as e:
                logger.error(f"Error registering tool {tool_name}: {str(e)}")
        elif tool_name not in function_tool_names:
            logger.debug(
                f"Tool '{tool_name}' is not a known system tool, will be handled later if appropriate"
            )

    logger.info(
        f"System registration complete: Registered {registered_count} system tools with configuration"
    )
    return registered_count


def copy_tool_from_source_to_target(
    source_registry: ToolRegistry, target_registry: ToolRegistry, tool_name: str
) -> bool:
    """Copy a tool from source registry to target registry.

    Args:
        source_registry: The source registry containing the tool
        target_registry: The target registry to copy the tool to
        tool_name: The name of the tool to copy

    Returns:
        True if registration succeeded, False otherwise
    """
    success, handler, definition = extract_tool_components(source_registry, tool_name)
    if not success:
        logger.error(f"Failed to extract components for tool {tool_name}")
        return False

    # Check if tool is already registered in target registry to avoid duplicates
    if tool_name in target_registry.tool_handlers:
        logger.debug(
            f"Tool {tool_name} already registered in target registry, skipping"
        )
        return True

    # Register the tool with the target registry
    target_registry.register_tool(tool_name, handler, definition)
    logger.debug(f"Registered tool {tool_name} in target registry")
    return True


def register_fork_tool(
    source_registry: ToolRegistry, target_registry: ToolRegistry, tool_name: str
) -> bool:
    """Register the fork tool with a runtime context.

    Args:
        source_registry: The source registry containing the tool
        target_registry: The target registry to register the tool in
        tool_name: The name of the tool to register

    Returns:
        True if registration succeeded, False otherwise
    """
    return copy_tool_from_source_to_target(source_registry, target_registry, tool_name)


def register_goto_tool(
    source_registry: ToolRegistry, target_registry: ToolRegistry, tool_name: str
) -> bool:
    """Register the goto tool with a runtime context.

    Args:
        source_registry: The source registry containing the tool
        target_registry: The target registry to register the tool in
        tool_name: The name of the tool to register

    Returns:
        True if registration succeeded, False otherwise
    """
    return copy_tool_from_source_to_target(source_registry, target_registry, tool_name)


def register_spawn_tool(
    source_registry: ToolRegistry,
    target_registry: ToolRegistry,
    tool_name: str,
    linked_programs: dict[str, Any],
    linked_program_descriptions: dict[str, str],
) -> bool:
    """Register the spawn tool with a runtime context.

    Args:
        source_registry: The source registry containing the tool
        target_registry: The target registry to register the tool in
        tool_name: The name of the tool to register
        linked_programs: Dictionary of linked programs
        linked_program_descriptions: Dictionary of program descriptions

    Returns:
        True if registration succeeded, False otherwise
    """
    # First get the base tool definition
    success, handler, schema = extract_tool_components(source_registry, tool_name)

    if not success:
        logger.warning(f"Could not extract components for {tool_name}")
        return False

    # Add available program descriptions to the tool description
    if linked_programs:
        # Build a list of available programs with descriptions
        available_programs_list = []

        # Include all programs with descriptions if available
        if linked_program_descriptions:
            for name, description in linked_program_descriptions.items():
                if name in linked_programs:
                    available_programs_list.append(f"'{name}': {description}")

        # Add any programs without descriptions
        for name in linked_programs:
            if not (
                linked_program_descriptions and name in linked_program_descriptions
            ):
                available_programs_list.append(f"'{name}'")

        # Format the list with a header
        if available_programs_list:
            formatted_programs = "\n\n## Available Programs:\n- " + "\n- ".join(
                available_programs_list
            )
            schema["description"] += formatted_programs

    # Register the customized tool
    target_registry.register_tool(tool_name, handler, schema)
    return True


def register_fd_tool(
    source_registry: ToolRegistry,
    target_registry: ToolRegistry,
    tool_name: str,
    fd_manager: Any,
) -> bool:
    """Register a file descriptor tool (read_fd or fd_to_file).

    Args:
        source_registry: The source registry containing the tool
        target_registry: The target registry to register the tool in
        tool_name: The name of the tool to register
        fd_manager: The file descriptor manager instance

    Returns:
        True if registration succeeded, False otherwise
    """
    # Check if dependencies are satisfied
    if not fd_manager:
        logger.warning(
            f"Cannot register {tool_name} tool: No file descriptor manager provided"
        )
        return False

    # Register the tool with fd_manager
    fd_manager.register_fd_tool(tool_name)

    # Copy from source to target
    return copy_tool_from_source_to_target(source_registry, target_registry, tool_name)
