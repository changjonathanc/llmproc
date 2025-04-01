"""Function-based tools for LLMProcess.

This module provides utilities for converting Python functions to LLM tools.
It handles extracting schemas from function signatures and docstrings,
converting Python types to JSON schema, and adapting functions to the tool interface.
"""

import asyncio
import functools
import inspect
import logging
import re
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, get_args, 
    get_origin, get_type_hints
)

from llmproc.tools.tool_result import ToolResult

# Set up logger
logger = logging.getLogger(__name__)


def register_tool(name: str = None, description: str = None, param_descriptions: Dict[str, str] = None):
    """Decorator to register a function as a tool.
    
    Args:
        name: Optional custom name for the tool (defaults to function name)
        description: Optional custom description for the tool (defaults to docstring)
        param_descriptions: Optional dict mapping parameter names to descriptions
            (overrides descriptions parsed from docstrings)
        
    Returns:
        Decorator function that registers the tool metadata
        
    Example:
        ```python
        @register_tool(
            name="weather_info", 
            description="Get weather for a location",
            param_descriptions={
                "location": "City name or postal code to get weather for",
                "units": "Temperature units (celsius or fahrenheit)"
            }
        )
        def get_weather(location: str, units: str = "celsius"):
            '''Get weather for a location.'''
            # Implementation...
            return {"temperature": 22, "units": units}
        ```
    """
    def decorator(func):
        # Store tool metadata as attributes on the function
        if name is not None:
            func._tool_name = name
        if description is not None:
            func._tool_description = description
        if param_descriptions is not None:
            func._param_descriptions = param_descriptions
        # Mark the function as a tool
        func._is_tool = True
        return func
    return decorator


def extract_docstring_params(func: Callable) -> Dict[str, Dict[str, str]]:
    """Extract parameter descriptions from a function's docstring.
    
    Args:
        func: The function to extract parameter descriptions from
        
    Returns:
        Dictionary mapping parameter names to their descriptions
    """
    # Get the docstring
    docstring = inspect.getdoc(func)
    if not docstring:
        return {}
        
    # Parameters extracted from docstring
    params = {}
    
    # Extract parameter descriptions from Args section
    args_match = re.search(r'Args:(.*?)(?:\n\n|\n\w+:|\Z)', docstring, re.DOTALL)
    if args_match:
        args_text = args_match.group(1)
        # Find all parameter descriptions
        param_matches = re.finditer(r'\n\s+(\w+):\s*(.*?)(?=\n\s+\w+:|$)', args_text, re.DOTALL)
        for match in param_matches:
            param_name = match.group(1)
            param_desc = match.group(2).strip()
            params[param_name] = {"description": param_desc}
            
    # Extract return description
    returns_match = re.search(r'Returns:(.*?)(?:\n\n|\n\w+:|\Z)', docstring, re.DOTALL)
    if returns_match:
        return_desc = returns_match.group(1).strip()
        params["return"] = {"description": return_desc}
        
    return params


def type_to_json_schema(
    type_hint: Any, 
    param_name: str, 
    docstring_params: Dict[str, Dict[str, str]],
    explicit_descriptions: Dict[str, str] = None
) -> Dict[str, Any]:
    """Convert a Python type hint to a JSON schema type.
    
    Args:
        type_hint: The Python type hint
        param_name: The parameter name (for documentation)
        docstring_params: Extracted parameter documentation
        explicit_descriptions: Optional explicit parameter descriptions that override docstring
        
    Returns:
        JSON schema representation of the type
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
                return type_to_json_schema(non_none_args[0], param_name, docstring_params)
                
    # Handle basic types
    if type_hint is str:
        schema["type"] = "string"
    elif type_hint is int:
        schema["type"] = "integer"
    elif type_hint is float:
        schema["type"] = "number"
    elif type_hint is bool:
        schema["type"] = "boolean"
        
    # Handle List[T]
    elif origin is list or type_hint is List or origin is List:
        schema["type"] = "array"
        # Get the item type if available
        if get_args(type_hint):
            item_type = get_args(type_hint)[0]
            # Convert the item type
            schema["items"] = type_to_json_schema(item_type, f"{param_name}_item", {})
        
    # Handle Dict[K, V]
    elif origin is dict or type_hint is Dict or origin is Dict:
        schema["type"] = "object"
        # We could further specify properties if we had more type info,
        # but for now we'll leave it as a generic object
    
    # Handle Any type
    elif type_hint is Any:
        # Allow any type
        del schema["type"]
        
    return schema


def function_to_tool_schema(func: Callable) -> Dict[str, Any]:
    """Convert a function to a tool schema.
    
    Args:
        func: The function to convert
        
    Returns:
        Tool schema compatible with LLMProcess
    """
    # Get function metadata
    func_name = getattr(func, "_tool_name", func.__name__)
    
    # Start with the basic schema
    schema = {
        "name": func_name,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
    
    # Get the docstring for the function
    docstring = inspect.getdoc(func)
    
    # Set description from tool metadata or function docstring
    if hasattr(func, "_tool_description"):
        schema["description"] = func._tool_description
    elif docstring:
        # Extract the first line of the docstring as the description
        first_line = docstring.split('\n', 1)[0].strip()
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
    
    # Add parameters to the schema
    for param_name, param in sig.parameters.items():
        # Skip self or cls params for class methods
        if param_name in ('self', 'cls'):
            continue
            
        # Get parameter type
        param_type = type_hints.get(param_name, Any)
        
        # Convert the type to JSON schema
        param_schema = type_to_json_schema(
            param_type, 
            param_name, 
            docstring_params, 
            explicit_descriptions
        )
        
        # Add to properties
        schema["input_schema"]["properties"][param_name] = param_schema
        
        # Add to required list if no default value
        if param.default is param.empty:
            schema["input_schema"]["required"].append(param_name)
    
    return schema


def prepare_tool_handler(func: Callable) -> Callable:
    """Create a tool handler from a function.
    
    Args:
        func: The function to convert
        
    Returns:
        Async function that handles tool calls with proper error handling
    """
    # Check if function is already async
    is_async = asyncio.iscoroutinefunction(func)
    
    # Get the function signature
    sig = inspect.signature(func)
    
    # Create handler function
    async def handler(args: Dict[str, Any]) -> ToolResult:
        try:
            # Create kwargs based on function signature
            kwargs = {}
            for param_name, param in sig.parameters.items():
                # Skip self or cls params
                if param_name in ('self', 'cls'):
                    continue
                    
                # Check if parameter is required but not provided
                if param.default is param.empty and param_name not in args:
                    return ToolResult.from_error(f"Missing required parameter: {param_name}")
                    
                # Add parameter if provided
                if param_name in args:
                    kwargs[param_name] = args[param_name]
            
            # Call the function
            if is_async:
                # Call async function
                result = await func(**kwargs)
            else:
                # Call sync function
                result = func(**kwargs)
                
            # Return success result
            return ToolResult(content=result, is_error=False)
            
        except Exception as e:
            # Return error result
            return ToolResult.from_error(f"Error executing tool: {str(e)}")
    
    return handler


def create_process_aware_handler(func: Callable, process: Any) -> Callable:
    """Create a process-aware tool handler that extracts args from a dictionary.
    
    This helps standardize the creation of system tool handlers that require
    an LLMProcess instance.
    
    Args:
        func: The tool function to wrap
        process: The LLMProcess instance to pass to the function
        
    Returns:
        Async function that extracts args and calls the tool with the process
    """
    # Get the function signature
    sig = inspect.signature(func)
    
    # Create a handler that will extract args and call the function
    async def handler(args: Dict[str, Any]) -> Any:
        # Gather kwargs from args dict based on parameter names
        kwargs = {}
        for param_name in sig.parameters:
            # Skip the llm_process parameter - we'll add it separately
            if param_name == 'llm_process':
                continue
                
            # Add parameter if provided
            if param_name in args:
                kwargs[param_name] = args[param_name]
        
        # Add the process parameter 
        kwargs["llm_process"] = process
        
        # Call the function (works for both sync and async)
        if asyncio.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            return func(**kwargs)
            
    return handler

def create_tool_from_function(func: Callable) -> Tuple[Callable, Dict[str, Any]]:
    """Create a complete tool (handler and schema) from a function.
    
    Args:
        func: The function to convert
        
    Returns:
        Tuple of (handler, schema) for the tool
    """
    # Generate the schema
    schema = function_to_tool_schema(func)
    
    # Prepare the handler
    handler = prepare_tool_handler(func)
    
    # Return both
    return handler, schema