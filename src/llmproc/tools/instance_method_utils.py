"""Utilities for handling instance methods in tool registration.

This module provides utilities for wrapping and handling instance methods
to make them compatible with the tool registration system.
"""

import asyncio
import functools
from collections.abc import Callable


def wrap_instance_method(method: Callable) -> Callable:
    """Convert an instance method to a standalone function that can hold metadata.

    This function creates a thin wrapper around an instance method that preserves
    all behavior but can have attributes set on it (unlike bound methods).

    Args:
        method: A bound instance method

    Returns:
        A standalone function that calls the original method

    Note:
        This function is primarily used internally by the register_tool decorator
        to support instance methods. The wrapper is not intended for direct use.
    """
    if not (hasattr(method, "__self__") and method.__self__ is not None):
        # Not a bound method, return as is
        return method

    # Get original information
    is_async = asyncio.iscoroutinefunction(method)
    instance = method.__self__
    method_name = method.__name__

    # Create appropriate wrapper based on sync/async
    if is_async:

        @functools.wraps(method)
        async def method_wrapper(*args, **kwargs):
            # Get method from instance to ensure proper binding
            bound_method = getattr(instance, method_name)
            return await bound_method(*args, **kwargs)
    else:

        @functools.wraps(method)
        def method_wrapper(*args, **kwargs):
            # Get method from instance to ensure proper binding
            bound_method = getattr(instance, method_name)
            return bound_method(*args, **kwargs)

    # Add metadata for clarity and debugging
    method_wrapper.__wrapped_instance_method__ = True
    method_wrapper.__original_instance__ = instance
    method_wrapper.__original_method_name__ = method_name

    return method_wrapper


def is_bound_method(func: Callable) -> bool:
    """Check if a callable is a bound instance method.

    Args:
        func: The callable to check

    Returns:
        True if the callable is a bound instance method
    """
    return hasattr(func, "__self__") and func.__self__ is not None
