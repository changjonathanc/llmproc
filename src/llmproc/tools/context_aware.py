"""Context-aware tool decorator for LLMProcess.

This module provides decorators for marking tools that require runtime context
access during execution. This enables dependency injection for tools without
creating circular dependencies at initialization time.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any, Optional, TypeVar

# Type variables for handler functions
F = TypeVar("F", bound=Callable[..., Any])


def context_aware(func: F) -> F:
    """Decorator marking a tool handler as requiring runtime context.

    This decorator explicitly marks tools that require access to the runtime context
    during execution, making the dependency requirement clear and intentional.

    Args:
        func: The tool handler function to mark as context-aware

    Returns:
        The decorated function with _needs_context attribute set to True

    Usage:
        @context_aware
        async def explicit_param_tool(param1, param2, runtime_context=None):
            process = runtime_context.get("process")
            # Use process and other context components
    """
    # Mark the function as needing context
    func._needs_context = True  # type: ignore

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Just pass through to the original function
        return await func(*args, **kwargs)

    # Preserve the _needs_context marker on the wrapper
    wrapper._needs_context = True  # type: ignore

    return wrapper  # type: ignore


def is_context_aware(handler: Callable) -> bool:
    """Check if a handler is marked as context-aware.

    Args:
        handler: The handler function to check

    Returns:
        True if the handler is marked as context-aware, False otherwise
    """
    return getattr(handler, "_needs_context", False)
