# Function-Based Tools

LLMProc supports registering Python functions as tools with automatic schema generation from type hints and docstrings. This provides a simple and intuitive way to create tools without writing boilerplate tool definition code.

## Basic Usage

```python
from llmproc import LLMProgram, register_tool

# Simple function with type hints
def get_calculator(x: int, y: int) -> int:
    """Calculate the sum of two numbers.
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        The sum of x and y
    """
    return x + y

# Create a program with a function tool
program = (
    LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant."
    )
    .add_tool(get_calculator)
    .compile()
)

# Start the LLM process
process = await program.start()
```

## Using the `register_tool` Decorator

For more control over tool names, descriptions, and parameter descriptions, use the `register_tool` decorator:

```python
from typing import Dict, Any
from llmproc import register_tool

@register_tool(
    name="weather_info", 
    description="Get weather information for a location",
    param_descriptions={
        "location": "City name or postal code to get weather for. More specific locations yield better results.",
        "units": "Temperature units to use in the response (either 'celsius' or 'fahrenheit')."
    }
)
def get_weather(location: str, units: str = "celsius") -> Dict[str, Any]:
    """Get weather for a location."""
    # Implementation...
    return {
        "location": location,
        "temperature": 22,
        "units": units,
        "conditions": "Sunny"
    }
```

The `param_descriptions` argument allows you to explicitly define parameter descriptions instead of relying on docstring parsing, which should be considered a fallback mechanism. Explicit parameter descriptions provide more control and clarity in your tool schemas.

## Async Function Support

Asynchronous functions are fully supported:

```python
import asyncio
from typing import Dict, Any
from llmproc import register_tool

@register_tool()
async def fetch_data(url: str, timeout: int = 30) -> Dict[str, Any]:
    """Fetch data from a URL.
    
    Args:
        url: The URL to fetch data from
        timeout: Request timeout in seconds
        
    Returns:
        The fetched data
    """
    # Async implementation
    await asyncio.sleep(0.1)  # Simulate network request
    return {
        "url": url,
        "data": f"Data from {url}",
        "status": 200
    }
```

## Type Hint Support

Function-based tools support automatic conversion of Python type hints to JSON Schema:

- Basic types: `str`, `int`, `float`, `bool`
- Complex types: `List[T]`, `Dict[K, V]`
- Optional types: `Optional[T]` (equivalent to `Union[T, None]`)
- Default values: Parameters with default values are marked as optional

## Docstring Parsing

The tool system automatically extracts parameter descriptions and return type information from Google-style docstrings:

```python
def search_documents(query: str, limit: int = 5):
    """Search documents by query.
    
    Args:
        query: The search query string
        limit: Maximum number of results to return
        
    Returns:
        List of document dictionaries matching the query
    """
    # Implementation...
```

## Fluent API Integration

Function-based tools integrate seamlessly with the fluent API:

```python
# Method chaining with multiple tools
program = (
    LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant."
    )
    .add_tool(get_calculator)
    .add_tool(get_weather)
    .add_tool(fetch_data)
    .preload_file("context.txt")
    .link_program("expert", expert_program, "A specialized expert program")
    .compile()
)
```

## Mixed Tool Types

You can mix function-based tools with dictionary-based tool configurations:

```python
# Add both function and dictionary tools
program = (
    LLMProgram(...)
    .add_tool(get_calculator)
    .add_tool({"name": "read_file", "enabled": True})
    .compile()
)
```

## Tool Error Handling

Tool errors are automatically handled and returned as proper error responses:

```python
def division_tool(x: int, y: int) -> float:
    """Divide two numbers.
    
    Args:
        x: Numerator
        y: Denominator
        
    Returns:
        The result of x / y
    """
    return x / y  # Will raise ZeroDivisionError if y is 0
```

When the LLM tries to call this tool with `y=0`, it will receive a proper error message indicating the division by zero error, rather than crashing the application.

## Initialization

Function tools are processed during program compilation. The `compile()` method:

1. Extracts schema information from type hints and docstrings
2. Creates async-compatible tool handlers
3. Registers tools with the tool registry

When the process is started, the tools are ready to use by the LLM.