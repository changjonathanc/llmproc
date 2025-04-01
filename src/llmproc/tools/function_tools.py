"""Function-based tool registration for LLMProcess.

This module provides utilities for registering Python functions as tools
for LLM processes, including automatic schema generation from type hints.
"""

import inspect
import re
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints, get_origin, get_args
from typing import TypedDict, Literal, Annotated
from docstring_parser import parse as parse_docstring

from llmproc.tools import ToolSchema, ToolHandler, ToolResult


def register_tool(
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Callable:
    """Decorator for registering a function as a tool.
    
    Args:
        name: Optional name override for the tool (defaults to function name)
        description: Optional description override (defaults to function docstring)
        
    Returns:
        Decorator function that processes and registers the tool
    """
    def decorator(func: Callable) -> Callable:
        # Store tool metadata in the function object
        func._tool_name = name or func.__name__
        func._tool_description = description or (func.__doc__ or "").strip()
        func._is_tool = True
        
        return func
    return decorator


def type_to_json_schema(type_hint: Type, param_name: str, docstring_params: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Convert a Python type hint to a JSON schema.
    
    Args:
        type_hint: The Python type hint to convert
        param_name: The name of the parameter for documentation lookup
        docstring_params: Dictionary of parameter documentation extracted from docstring
        
    Returns:
        JSON schema representation of the type
    """
    # Base schema with description from docstring if available
    schema = {}
    if param_name in docstring_params:
        schema["description"] = docstring_params[param_name].get("description", "")
    
    # Handle None or missing type
    if type_hint is None:
        schema["type"] = "object"
        return schema
    
    # Handle basic types
    if type_hint is str:
        schema["type"] = "string"
    elif type_hint is int:
        schema["type"] = "integer"
    elif type_hint is float:
        schema["type"] = "number"
    elif type_hint is bool:
        schema["type"] = "boolean"
    elif type_hint is dict or type_hint is Dict:
        schema["type"] = "object"
    elif type_hint is list or type_hint is List:
        schema["type"] = "array"
        schema["items"] = {}  # Default to any type of items
    
    # Handle optional types (Union[Type, None] or Optional[Type])
    elif get_origin(type_hint) is Union:
        args = get_args(type_hint)
        # Check if this is Optional[Type] (Union[Type, None])
        if type(None) in args:
            # Create schema for the non-None type
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                # Simple Optional[Type]
                main_schema = type_to_json_schema(non_none_args[0], param_name, docstring_params)
                schema.update(main_schema)
            else:
                # More complex union that includes None
                schema["anyOf"] = [
                    type_to_json_schema(arg, param_name, docstring_params) 
                    for arg in non_none_args
                ]
        else:
            # Regular Union[Type1, Type2, ...]
            schema["anyOf"] = [
                type_to_json_schema(arg, param_name, docstring_params) 
                for arg in args
            ]
    
    # Handle generic types with type arguments (e.g., List[str], Dict[str, int])
    elif get_origin(type_hint) is not None:
        origin = get_origin(type_hint)
        args = get_args(type_hint)
        
        if origin is list or origin is List:
            schema["type"] = "array"
            if args:
                schema["items"] = type_to_json_schema(args[0], f"{param_name}_item", docstring_params)
        
        elif origin is dict or origin is Dict:
            schema["type"] = "object"
            if len(args) >= 2:
                # We don't fully represent the key type, but we can represent the value type
                # For key types other than str, this is an approximation
                value_type = args[1]
                schema["additionalProperties"] = type_to_json_schema(
                    value_type, f"{param_name}_value", docstring_params
                )
        
        elif origin is Literal:
            # Handle Literal[val1, val2, ...]
            enum_values = list(args)
            schema["enum"] = enum_values
            # Determine the type based on the first value's type
            if enum_values and all(isinstance(val, str) for val in enum_values):
                schema["type"] = "string"
            elif enum_values and all(isinstance(val, (int, float)) for val in enum_values):
                schema["type"] = "number"
    
    # Default fallback
    else:
        schema["type"] = "object"
        schema["description"] = f"Complex type: {type_hint.__name__ if hasattr(type_hint, '__name__') else str(type_hint)}"
    
    return schema


def extract_docstring_params(func: Callable) -> Dict[str, Dict[str, str]]:
    """Extract parameter descriptions from function docstring.
    
    Args:
        func: The function to process
        
    Returns:
        Dictionary mapping parameter names to their descriptions
    """
    result = {}
    
    if not func.__doc__:
        return result
    
    # Parse the docstring
    try:
        docstring = parse_docstring(func.__doc__)
        
        # Extract parameter descriptions
        for param in docstring.params:
            result[param.arg_name] = {
                "description": param.description or "",
                "type": param.type_name or "",
            }
        
        # Store return type description
        if docstring.returns:
            result["return"] = {
                "description": docstring.returns.description or "",
                "type": docstring.returns.type_name or "",
            }
            
    except Exception as e:
        # Fallback to simple regex for basic extraction if docstring_parser fails
        param_pattern = r":param\s+(\w+):\s*(.*?)(?=$|:param\s+\w+:|:return:)"
        return_pattern = r":return:\s*(.*?)(?=$)"
        
        for match in re.finditer(param_pattern, func.__doc__, re.MULTILINE | re.DOTALL):
            param_name, description = match.groups()
            result[param_name] = {"description": description.strip(), "type": ""}
            
        return_match = re.search(return_pattern, func.__doc__, re.MULTILINE | re.DOTALL)
        if return_match:
            result["return"] = {"description": return_match.group(1).strip(), "type": ""}
            
    return result


def function_to_tool_schema(func: Callable) -> ToolSchema:
    """Convert a function to a tool schema based on type hints and docstring.
    
    Args:
        func: The function to convert
        
    Returns:
        Tool schema with name, description, and input parameters schema
    """
    # Get function name
    tool_name = getattr(func, "_tool_name", func.__name__)
    
    # Get function description from docstring or override
    full_docstring = func.__doc__ or ""
    # Extract the first paragraph from docstring as primary description
    first_paragraph = full_docstring.split("\n\n")[0].strip() if full_docstring else ""
    
    tool_description = getattr(func, "_tool_description", first_paragraph)
    
    # Extract all parameters from docstring
    docstring_params = extract_docstring_params(func)
    
    # Get parameters and types
    sig = inspect.signature(func)
    parameters = sig.parameters
    type_hints = get_type_hints(func)
    
    # Build the input schema
    properties = {}
    required = []
    
    for param_name, param in parameters.items():
        # Skip self/cls for methods
        if param_name in ("self", "cls"):
            continue
        
        # Get type hint or default to Any
        param_type = type_hints.get(param_name, Any)
        
        # Create schema for this parameter
        param_schema = type_to_json_schema(param_type, param_name, docstring_params)
        
        # Add parameter to properties
        properties[param_name] = param_schema
        
        # If no default value, it's required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    
    # Build the final schema
    input_schema = {
        "type": "object",
        "properties": properties,
        "required": required
    }
    
    # Create the complete tool schema
    schema: ToolSchema = {
        "name": tool_name,
        "description": tool_description,
        "input_schema": input_schema
    }
    
    return schema


def prepare_tool_handler(func: Callable) -> ToolHandler:
    """Create a tool handler from a function.
    
    Args:
        func: The function to convert to a tool handler
        
    Returns:
        An async function that acts as a tool handler, adapting based on whether
        the source function is sync or async
    """
    # Check if the function is already async
    is_async = inspect.iscoroutinefunction(func)
    
    # Create a handler function that properly adapts to the original function
    async def tool_handler(args: Dict[str, Any]) -> Any:
        try:
            # Extract arguments from the args dictionary based on parameter names
            sig = inspect.signature(func)
            call_args = {}
            
            for param_name, param in sig.parameters.items():
                # Skip self/cls for methods
                if param_name in ("self", "cls"):
                    continue
                
                # Get the value from args or use default
                if param_name in args:
                    call_args[param_name] = args[param_name]
                elif param.default is not inspect.Parameter.empty:
                    # Don't need to add param with default value if not specified
                    pass
                else:
                    # Missing required parameter
                    return ToolResult.from_error(f"Missing required parameter: {param_name}")
            
            # Call the function with the extracted arguments
            if is_async:
                # Direct await for async functions
                result = await func(**call_args)
            else:
                # Regular call for synchronous functions
                result = func(**call_args)
            
            # Handle result - convert to ToolResult if not already
            if isinstance(result, ToolResult):
                return result
            else:
                return ToolResult.from_success(result)
                
        except Exception as e:
            # Catch any errors during execution
            error_msg = f"Error executing tool: {str(e)}"
            return ToolResult.from_error(error_msg)
    
    return tool_handler


def create_tool_from_function(func: Callable) -> tuple[ToolHandler, ToolSchema]:
    """Create a complete tool from a function.
    
    Args:
        func: The function to convert to a tool
        
    Returns:
        A tuple of (handler, schema) for the tool
    """
    # Generate schema from function signature and docstring
    schema = function_to_tool_schema(func)
    
    # Create handler that adapts the function to the tool interface
    handler = prepare_tool_handler(func)
    
    return handler, schema