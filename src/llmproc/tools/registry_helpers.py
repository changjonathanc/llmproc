"""Helper functions for working with tool registries.

This module contains utility functions for working with ToolRegistry objects,
including copying tools between registries and checking for duplicate definitions.
"""

import logging
from typing import Any

# Set up logger
logger = logging.getLogger(__name__)


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
