# SDK Developer Experience

LLMProc provides a fluent, Pythonic SDK interface for creating and configuring LLM programs. This guide describes how to use the new SDK features implemented in RFC018.

## Fluent API

The fluent API allows for method chaining to create and configure LLM programs:

```python
from llmproc import LLMProgram

program = (
    LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a helpful assistant."
    )
    .add_tool(my_tool_function)
    .preload_file("context.txt")
    .link_program("expert", expert_program, "An expert program")
    .compile()
)

# Start the process
process = await program.start()
```

## Program Creation and Configuration

### Basic Initialization

```python
from llmproc import LLMProgram

# Create a basic program
program = LLMProgram(
    model_name="gpt-4",  
    provider="openai",
    system_prompt="You are a helpful assistant."
)
```

### Method Chaining

All configuration methods return `self` to allow for method chaining:

```python
# Configure a program with method chaining
program = (
    LLMProgram(...)
    .preload_file("file1.md")
    .preload_file("file2.md")
    .add_tool(tool_function)
    .compile()
)
```

### Program Linking

Link multiple specialized programs together:

```python
# Create specialized programs
math_program = LLMProgram(
    model_name="gpt-4",
    provider="openai",
    system_prompt="You are a math expert."
)

code_program = LLMProgram(
    model_name="claude-3-opus-20240229",
    provider="anthropic",
    system_prompt="You are a coding expert."
)

# Create a main program linked to the specialized programs
main_program = (
    LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a helpful assistant."
    )
    .link_program("math", math_program, "Expert in mathematics")
    .link_program("code", code_program, "Expert in coding")
    .compile()
)
```

### Compilation

All programs need to be compiled before starting:

```python
# Compile the program
program.compile()

# Start the process
process = await program.start()
```

You can also chain the compilation and starting:

```python
process = await program.compile().start()
```

## Function-Based Tools

LLMProc supports registering Python functions as tools with automatic schema generation from type hints and docstrings.

### Basic Function Tool

```python
from llmproc import LLMProgram

# Simple function with type hints
def add_numbers(x: int, y: int) -> int:
    """Calculate the sum of two numbers.
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        The sum of x and y
    """
    return x + y

# Create a program with the function tool
program = (
    LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a helpful assistant."
    )
    .add_tool(add_numbers)
    .compile()
)

# Start the process
process = await program.start()
```

### Using the `register_tool` Decorator

For more control over tool names and descriptions, use the `register_tool` decorator:

```python
from typing import Dict, Any
from llmproc import register_tool, LLMProgram

@register_tool(name="weather_info", description="Get weather information for a location")
def get_weather(location: str, units: str = "celsius") -> Dict[str, Any]:
    """Get weather for a location.
    
    Args:
        location: City or address
        units: Temperature units (celsius or fahrenheit)
        
    Returns:
        Weather information including temperature and conditions
    """
    # Implementation...
    return {
        "location": location,
        "temperature": 22,
        "units": units,
        "conditions": "Sunny"
    }

# Create a program with the decorated function tool
program = (
    LLMProgram(...)
    .add_tool(get_weather)
    .compile()
)
```

### Async Function Tools

Asynchronous functions are fully supported:

```python
import asyncio
from typing import Dict, Any
from llmproc import register_tool, LLMProgram

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

# Create a program with the async function tool
program = (
    LLMProgram(...)
    .add_tool(fetch_data)
    .compile()
)
```

### Type Conversion

Function-based tools support automatic conversion of Python type hints to JSON Schema:

- Basic types: `str`, `int`, `float`, `bool`
- Complex types: `List[T]`, `Dict[K, V]`
- Optional types: `Optional[T]` (equivalent to `Union[T, None]`)
- Default values: Parameters with default values are marked as optional

## Complete Example

Here's a complete example showing function-based tools with a fluent API:

```python
import asyncio
from typing import Dict, Any
from llmproc import LLMProgram, register_tool

# Define a calculator function
@register_tool(name="math_calculator", description="Perform arithmetic calculations")
def calculate(expression: str) -> Dict[str, Any]:
    """Calculate the result of an arithmetic expression.
    
    Args:
        expression: A mathematical expression like "2 + 2" or "5 * 10"
        
    Returns:
        A dictionary with the result and the parsed expression
    """
    # Simple evaluation (with proper safety checks in a real implementation)
    result = eval(expression, {"__builtins__": {}})
    return {
        "expression": expression,
        "result": result
    }

# Define a weather lookup function
@register_tool()
def weather_lookup(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """Look up weather information for a location.
    
    Args:
        location: City name or address
        unit: Temperature unit (celsius or fahrenheit)
        
    Returns:
        Weather information for the location
    """
    # Simulate weather lookup
    temps = {
        "New York": {"celsius": 22, "fahrenheit": 72},
        "London": {"celsius": 18, "fahrenheit": 64},
    }
    
    # Default to a moderate temperature if location not found
    temp = temps.get(location, {"celsius": 21, "fahrenheit": 70})
    
    return {
        "location": location,
        "temperature": temp[unit.lower()] if unit.lower() in temp else temp["celsius"],
        "unit": unit.lower(),
        "conditions": "Sunny",
        "humidity": "60%"
    }

async def main():
    # Set up callbacks to monitor tool usage
    def on_tool_start(tool_name, tool_args):
        print(f"Starting tool: {tool_name}")
        print(f"Arguments: {tool_args}")
        
    def on_tool_end(tool_name, result):
        print(f"Tool completed: {tool_name}")
        print(f"Result: {result.content}")
    
    callbacks = {
        "on_tool_start": on_tool_start,
        "on_tool_end": on_tool_end
    }
    
    # Create a program with function tools using the fluent API
    program = (
        LLMProgram(
            model_name="claude-3-haiku-20240307",
            provider="anthropic",
            system_prompt="You are a helpful assistant with tools.",
            parameters={"max_tokens": 1024}
        )
        .add_tool(calculate)
        .add_tool(weather_lookup)
        .compile()
    )
    
    # Start the process
    process = await program.start()
    
    # Run a query
    user_prompt = "What's the result of 125 * 48 and what's the weather in London?"
    result = await process.run(user_prompt, callbacks=callbacks)
    
    # Print the response
    print("\nFinal response:")
    print(process.get_last_message())

if __name__ == "__main__":
    asyncio.run(main())
```

## Type Hint Support

The function tool system automatically extracts:

1. **Parameter types** from type hints
2. **Parameter descriptions** from docstrings
3. **Default values** from function signatures
4. **Return types** from type hints
5. **Return descriptions** from docstrings

These are all converted to a JSON schema that the LLM can use to understand how to call the tool correctly.

## Docstring Format

The tool system works best with Google-style docstrings:

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

## Error Handling

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

If the LLM tries to call this tool with `y=0`, it will receive a proper error message indicating the division by zero error.

## Further Reading

- [Function-Based Tools](function-based-tools.md) - Detailed documentation on function-based tools
- See [examples/features/function_tools.py](../examples/features/function_tools.py) for a complete working example