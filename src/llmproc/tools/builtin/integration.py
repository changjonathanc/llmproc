"""Integration functions for builtin tools.

These functions are deprecated and maintained only for backward compatibility.
The ToolManager's initialize_tools method now provides direct registration
without the need for intermediate registries.

These will be removed in a future release.
"""

import logging
from collections.abc import Callable
from typing import Any, Optional

from llmproc.common.results import ToolResult
from llmproc.file_descriptors.constants import (
    FD_RELATED_TOOLS,
)
from llmproc.tools.builtin import BUILTIN_TOOLS

# Import builtin tool components
from llmproc.tools.builtin.calculator import calculator
from llmproc.tools.builtin.fd_tools import fd_to_file_tool, read_fd_tool
from llmproc.tools.builtin.fork import fork_tool
from llmproc.tools.builtin.goto import handle_goto
from llmproc.tools.builtin.list_dir import list_dir
from llmproc.tools.builtin.read_file import read_file
from llmproc.tools.builtin.spawn import spawn_tool
from llmproc.tools.function_tools import create_tool_from_function
from llmproc.tools.registry_data import get_function_tool_names
from llmproc.tools.registry_helpers import extract_tool_components
from llmproc.tools.tool_registry import ToolRegistry

# Set up logger
logger = logging.getLogger(__name__)


def load_builtin_tools(registry: ToolRegistry) -> bool:
    """Load all available builtin tools to the provided registry.

    DEPRECATED: This function is maintained only for backward compatibility.
    The ToolManager's initialize_tools method now provides direct registration.

    This is typically done once during initialization and doesn't depend on which tools are enabled.

    Args:
        registry: The registry to load builtin tools into

    Returns:
        True if registration succeeded, False otherwise
    """
    logger.info("Loading builtin tools into registry")

    # Import function tools utilities and builtin tools

    # Register each tool from the central mapping
    # Note: We don't pass config here since this is just initial catalog loading
    for name, func in BUILTIN_TOOLS.items():
        try:
            handler, definition = create_tool_from_function(func)
            registry.register_tool(name, handler, definition)
        except Exception as e:
            logger.error(f"Error registering builtin tool {name}: {str(e)}")

    logger.info(f"Finished loading {len(BUILTIN_TOOLS)} builtin tools into registry")
    return True


def register_system_tools(
    source_registry: ToolRegistry,
    target_registry: ToolRegistry,
    enabled_tools: list[str],
    config: dict[str, Any],
) -> int:
    """Register system tools based on configuration.

    DEPRECATED: This function is maintained only for backward compatibility.
    The ToolManager's initialize_tools method now provides direct registration
    without the need for intermediate registries.

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
    logger.info(f"Starting system tools registration based on enabled list: {enabled_tools}")

    # Import necessary components

    # Extract config components for dependency checking
    fd_manager = config.get("fd_manager")
    has_linked_programs = config.get("has_linked_programs", False)
    fd_enabled = fd_manager is not None

    # Get function-based tools from registry
    function_tool_names = get_function_tool_names()

    # Add the read_fd tool to enabled_tools if fd_manager is available but not enabled explicitly
    # We only add read_fd by default as it's the basic functionality, fd_to_file requires explicit opt-in
    if fd_enabled and "read_fd" not in enabled_tools:
        logger.info("File descriptor system is enabled, automatically enabling read_fd tool")
        enabled_tools.append("read_fd")

    registered_count = 0

    # Process each enabled tool
    for tool_name in enabled_tools:
        # Skip if not a builtin tool and not handled elsewhere
        if tool_name not in BUILTIN_TOOLS and tool_name not in function_tool_names:
            logger.debug(f"Tool '{tool_name}' is not a known system tool, will be handled later if appropriate")
            continue

        # Skip if it's not a builtin tool (will be handled elsewhere)
        if tool_name not in BUILTIN_TOOLS:
            continue

        # Basic dependency checks
        if tool_name == "spawn" and not has_linked_programs:
            logger.info(f"Skipping {tool_name} - no linked programs available")
            continue

        if tool_name in ("read_fd", "fd_to_file") and not fd_manager:
            logger.info(f"Skipping {tool_name} - no fd_manager available")
            continue

        try:
            # Get the function from the mapping
            func = BUILTIN_TOOLS[tool_name]

            # Create handler and schema with config for schema modifiers
            handler, schema = create_tool_from_function(func, config)

            # Register to target registry
            target_registry.register_tool(tool_name, handler, schema)
            registered_count += 1
            logger.debug(f"Successfully registered system tool: {tool_name}")
        except Exception as e:
            logger.error(f"Error registering tool {tool_name}: {str(e)}")

    logger.info(f"System registration complete: Registered {registered_count} system tools with configuration")
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
        logger.debug(f"Tool {tool_name} already registered in target registry, skipping")
        return True

    # Register the tool with the target registry
    target_registry.register_tool(tool_name, handler, definition)
    logger.debug(f"Registered tool {tool_name} in target registry")
    return True
