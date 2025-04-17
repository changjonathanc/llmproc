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
import re
from collections.abc import Callable
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

from llmproc.common.context import check_requires_context, validate_context_has
from llmproc.common.results import ToolResult


def get_tool_name(tool: Callable) -> str:
    """Extract tool name from a callable function.

    Args:
        tool: The callable tool function

    Returns:
        The name of the tool (from _tool_name attribute or function name)
    """
    return getattr(tool, "_tool_name", tool.__name__)


# Set up logger
logger = logging.getLogger(__name__)


def register_tool(
    name: str = None,
    description: str = None,
    param_descriptions: dict[str, str] = None,
    schema: dict[str, Any] = None,
    required: list[str] = None,
    requires_context: bool = False,
    required_context_keys: list[str] = None,
    schema_modifier: Callable[[dict, dict], dict] = None,
):
    """Decorator to register a function as a tool with enhanced schema support.

    Args:
        name: Optional custom name for the tool (defaults to function name)
        description: Optional custom description for the tool (defaults to docstring)
        param_descriptions: Optional dict mapping parameter names to descriptions
        schema: Optional custom JSON schema for the tool (overrides auto-generated schema)
        required: Optional list of required parameter names (overrides detection from signature)
        requires_context: Whether this tool requires runtime context
        required_context_keys: List of context keys that must be present in runtime_context
        schema_modifier: Optional function to modify schema with runtime config

    Returns:
        Decorator function that registers the tool metadata
    """
    # Handle case where decorator is used without arguments: @register_tool
    if callable(name):
        func = name
        func._is_tool = True
        return func

    def decorator(func):
        # Store tool metadata as attributes on the function
        if name is not None:
            func._tool_name = name
        if description is not None:
            func._tool_description = description
        if param_descriptions is not None:
            func._param_descriptions = param_descriptions
        if schema is not None:
            func._custom_schema = schema
        if required is not None:
            func._required_params = required
        if schema_modifier is not None:
            func._schema_modifier = schema_modifier

        # Mark the function as a tool
        func._is_tool = True

        # Handle context awareness
        if requires_context:
            # Mark the function as requiring context
            func._requires_context = True

            # Store required context keys if specified
            if required_context_keys:
                func._required_context_keys = required_context_keys

            # Create wrapper to validate context requirements
            @functools.wraps(func)
            async def context_wrapper(*args, **kwargs):
                # Extract function name once to avoid duplication
                func_name = getattr(func, "_tool_name", func.__name__)

                # Validate context if requirements specified
                if required_context_keys and "runtime_context" in kwargs:
                    valid, error = validate_context_has(kwargs["runtime_context"], *required_context_keys)
                    if not valid:
                        error_msg = f"Tool '{func_name}' error: {error}"
                        logger.error(error_msg)
                        return ToolResult.from_error(error_msg)

                # Execute with simplified error handling
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_msg = f"Tool '{func_name}' error: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return ToolResult.from_error(error_msg)

            # Copy all metadata to wrapper
            for attr in [
                "_is_tool",
                "_requires_context",
                "_tool_name",
                "_tool_description",
                "_param_descriptions",
                "_custom_schema",
                "_required_params",
                "_required_context_keys",
                "_schema_modifier",  # Add schema_modifier to copied attributes
            ]:
                if hasattr(func, attr):
                    setattr(context_wrapper, attr, getattr(func, attr))

            return context_wrapper

        return func

    return decorator


def extract_docstring_params(func: Callable) -> dict[str, dict[str, str]]:
    """Extract parameter descriptions from a function's docstring."""
    docstring = inspect.getdoc(func)
    if not docstring:
        return {}

    params = {}

    # Extract parameter descriptions from Args section
    args_match = re.search(r"Args:(.*?)(?:\n\n|\n\w+:|\Z)", docstring, re.DOTALL)
    if args_match:
        args_text = args_match.group(1)
        param_matches = re.finditer(r"\n\s+(\w+):\s*(.*?)(?=\n\s+\w+:|$)", args_text, re.DOTALL)
        for match in param_matches:
            param_name = match.group(1)
            param_desc = match.group(2).strip()
            params[param_name] = {"description": param_desc}

    # Extract return description
    returns_match = re.search(r"Returns:(.*?)(?:\n\n|\n\w+:|\Z)", docstring, re.DOTALL)
    if returns_match:
        return_desc = returns_match.group(1).strip()
        params["return"] = {"description": return_desc}

    return params


def type_to_json_schema(
    type_hint: Any,
    param_name: str,
    docstring_params: dict[str, dict[str, str]],
    explicit_descriptions: dict[str, str] = None,
) -> dict[str, Any]:
    """Convert a Python type hint to a JSON schema type."""
    # Start with a default schema
    schema = {"type": "string"}  # Default to string if we can't determine

    # Get description - prioritize explicit description over docstring
    if explicit_descriptions and param_name in explicit_descriptions:
        schema["description"] = explicit_descriptions[param_name]
    elif param_name in docstring_params:
        schema["description"] = docstring_params[param_name]["description"]

    # Handle Optional types (Union[T, None])
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        # Check if it's Optional (one of the args is NoneType)
        if type(None) in args:
            # Get the non-None type
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                # Convert the non-None type
                return type_to_json_schema(non_none_args[0], param_name, docstring_params, explicit_descriptions)

    # Handle basic types
    if type_hint is str:
        schema["type"] = "string"
    elif type_hint is int:
        schema["type"] = "integer"
    elif type_hint is float:
        schema["type"] = "number"
    elif type_hint is bool:
        schema["type"] = "boolean"
    # Handle list[T]
    elif origin is list or type_hint is list:
        schema["type"] = "array"
        # Get the item type if available
        if get_args(type_hint):
            item_type = get_args(type_hint)[0]
            # Convert the item type
            schema["items"] = type_to_json_schema(item_type, f"{param_name}_item", {}, explicit_descriptions)
    # Handle dict[K, V]
    elif origin is dict or type_hint is dict:
        schema["type"] = "object"
    # Handle Any type
    elif type_hint is Any:
        # Allow any type
        del schema["type"]

    return schema


def function_to_tool_schema(func: Callable) -> dict[str, Any]:
    """Convert a function to a tool schema."""
    # If there's a custom schema defined, just use that
    if hasattr(func, "_custom_schema"):
        return func._custom_schema

    # Get function metadata
    func_name = getattr(func, "_tool_name", func.__name__)

    # Start with the basic schema
    schema = {
        "name": func_name,
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }

    # Get the docstring for the function
    docstring = inspect.getdoc(func)

    # Set description from tool metadata or function docstring
    if hasattr(func, "_tool_description"):
        schema["description"] = func._tool_description
    elif docstring:
        # Extract the first line of the docstring as the description
        first_line = docstring.split("\n", 1)[0].strip()
        schema["description"] = first_line
    else:
        schema["description"] = f"Tool for {func_name}"

    # Extract parameter documentation from docstring
    docstring_params = extract_docstring_params(func)

    # Get explicit parameter descriptions if provided
    explicit_descriptions = getattr(func, "_param_descriptions", None)

    # Get type hints and signature
    type_hints = get_type_hints(func)
    sig = inspect.signature(func)

    # Build schema properties and required parameters in a single pass
    for param_name, param in sig.parameters.items():
        # Skip special parameters
        if param_name in ("self", "cls", "runtime_context"):
            continue

        # Get parameter type
        param_type = type_hints.get(param_name, Any)

        # Convert the type to JSON schema
        param_schema = type_to_json_schema(
            param_type, param_name, docstring_params, explicit_descriptions
        )

        # Add to properties
        schema["input_schema"]["properties"][param_name] = param_schema

        # Add to required list if no default value and no custom required list
        if not hasattr(func, "_required_params") and param.default is param.empty:
            schema["input_schema"]["required"].append(param_name)

    # Override with explicitly provided required parameters if specified
    if hasattr(func, "_required_params"):
        schema["input_schema"]["required"] = func._required_params

    return schema


def prepare_tool_handler(func: Callable) -> Callable:
    """Create a tool handler from a function with proper error handling."""
    # Check if function is already async
    is_async = asyncio.iscoroutinefunction(func)

    # Get the function signature
    sig = inspect.signature(func)

    # Get the tool name for error reporting
    func_name = getattr(func, "_tool_name", func.__name__)

    # Create handler function with error handling
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

            # Return success result
            return ToolResult(content=result, is_error=False)

        except Exception as e:
            # Return error result
            error_msg = f"Tool '{func_name}' error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolResult.from_error(error_msg)

    # Copy metadata attributes
    for attr in ["_requires_context", "_required_context_keys"]:
        if hasattr(func, attr):
            setattr(handler, attr, getattr(func, attr))

    return handler


def create_process_aware_handler(func: Callable, process: Any) -> Callable:
    """Create a process-aware tool handler that injects the process instance."""
    # Get the function signature
    sig = inspect.signature(func)

    # Use a set for faster lookups
    param_names = {name for name in sig.parameters if name != "llm_process"}

    # Create a handler with process injection
    async def handler(**kwargs) -> Any:
        # Filter kwargs to only include those in the function signature
        function_kwargs = {k: v for k, v in kwargs.items() if k in param_names}

        # Add the process parameter
        function_kwargs["llm_process"] = process

        # Call the function (works for both sync and async)
        if asyncio.iscoroutinefunction(func):
            return await func(**function_kwargs)
        else:
            return func(**function_kwargs)

    return handler


def create_tool_from_function(func: Callable, config: dict = None) -> tuple[Callable, dict[str, Any]]:
    """Create a complete tool (handler and schema) from a function.

    Args:
        func: The function to create a tool from
        config: Optional configuration dictionary for schema modification

    Returns:
        Tuple of (handler, schema)
    """
    handler = prepare_tool_handler(func)
    schema = function_to_tool_schema(func)

    # Apply schema modifier if present and config is provided
    if config and hasattr(func, "_schema_modifier"):
        schema = func._schema_modifier(schema, config)

    return handler, schema
