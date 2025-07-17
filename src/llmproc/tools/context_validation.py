"""Context validation utilities for tool management.

This module provides helper functions for validating and managing runtime context
for tools that require context-aware execution.
"""

import logging
from collections.abc import Callable
from typing import Any, Optional

from llmproc.common.context import RuntimeContext, validate_context_has
from llmproc.common.metadata import get_tool_meta

logger = logging.getLogger(__name__)


def validate_context_for_tool(
    tool_name: str, handler: Callable, runtime_context: RuntimeContext
) -> tuple[bool, Optional[str]]:
    """Validate that the runtime context has required keys for a tool.

    Args:
        tool_name: Name of the tool being validated
        handler: The tool handler function
        runtime_context: The runtime context to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    meta = get_tool_meta(handler)

    if not meta.requires_context:
        return True, None

    if not runtime_context or "process" not in runtime_context:
        return False, f"Tool '{tool_name}' requires runtime context with 'process'"

    valid, error = validate_context_has(runtime_context, "process")
    if not valid:
        return False, f"Tool '{tool_name}' context validation failed: {error}"

    return True, None


def prepare_arguments_with_context(args: dict[str, Any], runtime_context: RuntimeContext) -> dict[str, Any]:
    """Prepare tool arguments by adding runtime context if needed.

    Args:
        args: Original tool arguments
        runtime_context: Runtime context to add

    Returns:
        Updated arguments dictionary with context added
    """
    # Add runtime context to arguments if available
    prepared_args = args.copy()
    if runtime_context:
        prepared_args["runtime_context"] = runtime_context

    return prepared_args
