"""Function-based tools for LLMProcess.

This module provides utilities for converting Python functions to LLM tools.
It handles extracting schemas from function signatures and docstrings,
converting Python types to JSON schema, and adapting functions to the tool interface.

For detailed examples and documentation, see:
- docs/function-based-tools.md - Complete documentation
- examples/features/function_tools.py - Basic and advanced function tools
- examples/multiply_example.py - Simple function tool example
"""

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, Union

from llmproc.common.access_control import AccessLevel
from llmproc.common.metadata import (
    ToolMeta,
    attach_meta,
    get_tool_meta,
)
from llmproc.common.results import ToolResult, ensure_tool_result
from llmproc.tools.function_schemas import (
    create_schema_from_callable,  # noqa: F401 - re-exported
    extract_docstring_params,  # noqa: F401 - re-exported
    function_to_tool_schema,  # noqa: F401 - re-exported
)
from llmproc.tools.instance_method_utils import is_bound_method, wrap_instance_method

# Set up logger
logger = logging.getLogger(__name__)


def register_tool(
    name: str = None,
    description: str = None,
    param_descriptions: dict[str, str] = None,
    schema: dict[str, Any] = None,
    required: list[str] = None,
    requires_context: bool = False,
    schema_modifier: Callable[[dict, dict], dict] = None,
    access: Union[AccessLevel, str] = AccessLevel.WRITE,
):
    """Decorator to register a function as a tool with enhanced schema support.

    This decorator stores all tool metadata in a centralized ToolMeta object
    rather than as separate attributes on the function.

    Args:
        name: Optional custom name for the tool (defaults to function name)
        description: Optional custom description for the tool (defaults to docstring)
        param_descriptions: Optional dict mapping parameter names to descriptions
        schema: Optional custom JSON schema for the tool (overrides auto-generated schema)
        required: Optional list of required parameter names (overrides detection from signature)
        requires_context: Whether this tool requires runtime context (process is always provided)
        schema_modifier: Optional function to modify schema with runtime config
        access: Access level for this tool (READ, WRITE, or ADMIN). Defaults to WRITE.

    Returns:
        Decorator function that registers the tool metadata
    """
    # Handle case where decorator is used without parentheses: @register_tool
    if callable(name):
        func = name  # type: ignore[assignment]
        # Reuse the code path by calling the decorator with defaults
        return register_tool()(func)

    def _finalize_registration(func: Callable, meta: ToolMeta) -> Callable:
        """Finalize tool registration by attaching metadata."""
        attach_meta(func, meta)
        return func

    def decorator(func):
        if is_bound_method(func):
            func = wrap_instance_method(func)

        access_level = access
        if isinstance(access, str):
            access_level = AccessLevel.from_string(access)

        tool_name = name if name is not None else func.__name__

        meta_obj = ToolMeta(
            name=tool_name,
            description=description,
            param_descriptions=param_descriptions,
            required_params=tuple(required or ()),
            custom_schema=schema,
            access=access_level,
            requires_context=requires_context,
            schema_modifier=schema_modifier,
        )

        attach_meta(func, meta_obj)

        if (
            not is_bound_method(func)
            and inspect.isfunction(func)
            and hasattr(func, "__qualname__")
            and "." in func.__qualname__
        ):
            parts = func.__qualname__.split(".")
            if len(parts) >= 2 and parts[-2] != "<locals>":
                func._deferred_tool_registration = True
                return func

        return _finalize_registration(func, meta_obj)

    return decorator


def create_handler_from_function(func: Callable) -> Callable:
    """Create a tool handler from a function with proper error handling."""
    # Check if function is already async
    is_async = asyncio.iscoroutinefunction(func)

    # Get the function signature
    sig = inspect.signature(func)

    # Get metadata from the centralized metadata object
    meta = get_tool_meta(func)
    func_name = meta.name or func.__name__

    # Create handler function with error handling
    @functools.wraps(func)
    async def handler(**kwargs) -> ToolResult:
        try:
            # Process function parameters efficiently
            function_kwargs = {}

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue

                # Check if parameter is required but not provided
                if param.default is param.empty and param_name not in kwargs:
                    return ToolResult.from_error(f"Tool '{func_name}' error: Missing required parameter: {param_name}")

                # Add parameter if provided
                if param_name in kwargs:
                    function_kwargs[param_name] = kwargs[param_name]

            # Call the function (async or sync)
            result = await func(**function_kwargs) if is_async else func(**function_kwargs)

            # Allow functions to return ToolResult directly without double wrapping
            return ensure_tool_result(result)

        except Exception as e:
            # Return error result
            error_msg = f"Tool '{func_name}' error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolResult.from_error(error_msg)

    # Transfer the metadata to the handler
    attach_meta(handler, meta)

    return handler
