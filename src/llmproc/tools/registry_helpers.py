"""Helper functions for working with tool registries.

This module contains utility functions for working with ToolRegistry objects,
including copying tools between registries, applying aliases, and checking for duplicates.
"""

import logging
from collections.abc import Callable
from typing import Any, Optional

from llmproc.common.results import ToolResult
from llmproc.tools.tool_registry import ToolRegistry

# Set up logger
logger = logging.getLogger(__name__)


def copy_tool_from_source_to_target(
    source_registry: ToolRegistry, target_registry: ToolRegistry, tool_name: str
) -> bool:
    """Copy a tool from source registry to target registry without customization.

    Args:
        source_registry: The source registry containing the tool
        target_registry: The target registry to copy the tool to
        tool_name: The name of the tool to copy

    Returns:
        True if registration succeeded, False otherwise
    """
    # Extract all components from source
    success, handler, definition_copy = extract_tool_components(source_registry, tool_name)
    if not (success and handler and definition_copy):
        return False

    # Register with target registry
    target_registry.register_tool(tool_name, handler, definition_copy)
    logger.debug(f"Copied {tool_name} tool from source to target registry")
    return True


def extract_tool_components(
    registry: ToolRegistry, tool_name: str
) -> tuple[bool, Optional[Callable], Optional[dict[str, Any]]]:
    """Extract handler and definition for a tool from a registry.

    Args:
        registry: The registry containing the tool
        tool_name: The name of the tool to extract

    Returns:
        Tuple (bool, handler, definition copy):
            - Success status (True/False)
            - Handler function (or None if not found)
            - Copy of definition (or None if not found)
    """
    if tool_name not in registry.tool_handlers:
        logger.error(f"Handler for {tool_name} not found in registry")
        return False, None, None

    handler = registry.get_handler(tool_name)

    # Find the matching definition
    definition = next((d for d in registry.get_definitions() if d.get("name") == tool_name), None)
    if not definition:
        logger.error(f"Definition for {tool_name} not found in registry")
        return False, handler, None

    # Make a copy of the definition to avoid modifying the original
    definition_copy = definition.copy()

    return True, handler, definition_copy


def check_for_duplicate_schema_names(
    schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter out duplicate tool schemas, keeping only the first occurrence of each name.

    Args:
        schemas: List of tool schemas to check

    Returns:
        List of schemas with duplicates removed
    """
    seen_names = {}  # Track name -> index
    unique_schemas = []

    for i, schema in enumerate(schemas):
        name = schema.get("name", "")
        if name in seen_names:
            logger.warning(
                f"Duplicate tool name '{name}' found at indices {seen_names[name]} and {i}. Keeping only the first occurrence."
            )
        else:
            seen_names[name] = i
            unique_schemas.append(schema)

    return unique_schemas


def apply_aliases_to_schemas(schemas: list[dict[str, Any]], reverse_aliases: dict[str, str]) -> list[dict[str, Any]]:
    """Apply aliases to schemas where applicable.

    Args:
        schemas: List of tool schemas
        reverse_aliases: Dictionary mapping original names to aliases

    Returns:
        List of schemas with aliases applied
    """
    result = []
    for schema in schemas:
        original_name = schema.get("name", "")
        if original_name in reverse_aliases:
            # Create a copy with alias name
            aliased_schema = schema.copy()
            aliased_schema["name"] = reverse_aliases[original_name]
            result.append(aliased_schema)
        else:
            # Keep original schema
            result.append(schema)
    return result
