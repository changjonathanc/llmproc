"""Type conversion utilities for tool schema generation.

This module provides utilities for converting Python type hints to JSON schema
representations, supporting the tool registration system.
"""

from typing import Any, Union, get_args, get_origin


def type_to_json_schema(
    type_hint: Any,
    param_name: str,
    docstring_params: dict[str, dict[str, str]],
    explicit_descriptions: dict[str, str] = None,
) -> dict[str, Any]:
    """Convert a Python type hint to a JSON schema type.

    Args:
        type_hint: The Python type hint to convert
        param_name: Name of the parameter (for description lookup)
        docstring_params: Parameter descriptions extracted from docstring
        explicit_descriptions: Explicit parameter descriptions to override docstring

    Returns:
        JSON schema dictionary for the type
    """
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
