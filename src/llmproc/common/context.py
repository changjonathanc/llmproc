"""Runtime context management for dependency injection.

This module provides type definitions, validation utilities, helper functions,
and decorators for working with runtime context for dependency injection in tools.
"""

import logging
from collections.abc import Callable

# Runtime context utilities live in common; avoid importing higher-level
# packages to keep layering clean.
from typing import Any, Optional, TypedDict, TypeVar

# Set up logger
logger = logging.getLogger(__name__)

# Type definitions
F = TypeVar("F", bound=Callable[..., Any])


class RuntimeContext(TypedDict, total=False):
    """Runtime context for dependency injection in tools.

    This TypedDict defines the standard structure for runtime context
    passed to tools that require runtime context. The 'total=False' means
    that not all keys are required in every context instance.
    """

    process: Any  # LLMProcess instance
    fd_manager: Any  # FileDescriptorManager instance
    linked_programs: dict[str, Any]  # Dictionary of linked programs
    linked_program_descriptions: dict[str, str]  # Dictionary of program descriptions
    stderr: list[str]  # Buffer for stderr logging via write_stderr tool


def validate_context_has(context: Optional[dict[str, Any]], *keys: str) -> tuple[bool, Optional[str]]:
    """Validate that context exists and has the required keys.

    Args:
        context: The runtime context to validate
        *keys: The required keys that should be present in the context

    Returns:
        Tuple of (valid, error_message):
          - valid: True if the context is valid, False otherwise
          - error_message: None if valid, otherwise a descriptive error message
    """
    # If no context at all (None), that's an error
    if context is None:
        return False, "Runtime context is missing"

    # Empty dictionary is valid if we're not checking for specific keys
    # If no keys were specified, just verify context is a dictionary
    if not keys:
        return True, None

    # Check for missing required keys
    missing = [key for key in keys if key not in context]
    if missing:
        return False, f"Runtime context missing required keys: {', '.join(missing)}"

    return True, None
