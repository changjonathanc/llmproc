# Function-Based Tools

LLMProc allows you to register Python functions as tools with minimal effort. The system automatically generates schemas from type hints and docstrings, eliminating the need for boilerplate code.

## Quick Start

```python
from llmproc import LLMProgram

# Define a simple function with type hints and docstring
def add_numbers(x: int, y: int) -> int:
    """Add two numbers together.

    Args:
        x: First number
        y: Second number

    Returns:
        The sum of x and y
    """
    return x + y

# Create a program and register the function as a tool
program = LLMProgram(
    model_name="claude-3-7-sonnet",
    provider="anthropic",
    system_prompt="You are a helpful assistant.",
    tools=[add_numbers]  # Simply pass your function
)

# Or set tools after creation
# program.register_tools([calculator, get_calculator, "read_file"])

# Start the LLM process
process = await program.start()
```

That's it! Your function is now available as a tool for the LLM to use.

## Core Concepts

### How It Works

When you register a function as a tool, LLMProc:

1. Extracts parameter types from type hints
2. Generates parameter descriptions from docstrings
3. Creates a JSON Schema for the LLM
4. Handles invocation and error management automatically

### Supported Function Types

You can register several types of functions as tools:

- Regular functions
- Async functions
- Class instance methods (both sync and async)
- Built-in tools (by name)

## Creating Function Tools

### Basic Function Tools

The simplest way to create a tool is to define a function with type hints:

```python
def search_documents(query: str, limit: int = 5) -> list:
    """Search documents by query.

    Args:
        query: The search query string
        limit: Maximum number of results to return

    Returns:
        List of document dictionaries matching the query
    """
    # Implementation...
    return [{"title": "Document 1", "content": "..."}, ...]
```

### Async Function Tools

Asynchronous functions work the same way:

```python
async def fetch_data(url: str, timeout: int = 30) -> dict:
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

### Class Instance Methods as Tools

You can use instance methods to create stateful tools:

```python
class DataProvider:
    def __init__(self):
        self.counter = 0
        self.data = {"users": ["Alice", "Bob"]}

    def get_data(self, key: str) -> dict:
        """Get data for the specified key.

        Args:
            key: The data key to retrieve

        Returns:
            Data associated with the key
        """
        self.counter += 1
        return {
            "key": key,
            "value": self.data.get(key, None),
            "access_count": self.counter
        }

# Create an instance and register the method
provider = DataProvider()
program.register_tools([provider.get_data])
```

When using instance methods:
- The method maintains access to instance state (`self`)
- The `self` parameter is automatically handled
- Both sync and async methods are supported

## Customizing Tools

### Using the `register_tool` Decorator

While entirely optional, the `register_tool` decorator gives you more control:

```python
from llmproc import register_tool

@register_tool(
    name="weather_info",                      # Custom tool name
    description="Get weather for a location", # Custom description
    param_descriptions={
        "location": "City name or postal code",
        "units": "Temperature units ('celsius' or 'fahrenheit')"
    }
)
def get_weather(location: str, units: str = "celsius") -> dict:
    """Get weather for a location."""
    # Implementation...
    return {
        "location": location,
        "temperature": 22,
        "units": units,
        "conditions": "Sunny"
    }
```

The decorator allows you to:
- Set a custom tool name
- Provide a detailed tool description
- Define explicit parameter descriptions
- Configure access control and context requirements

### Context-Aware Tools

You can create tools that access the LLMProcess runtime context:

```python
@register_tool(
    name="spawn_tool",
    description="Create a new process from a linked program",
    requires_context=True,  # Mark tool as requiring runtime context
    required_context_keys=["process"]  # Specify required context keys
)
async def spawn_child_process(
    program_name: str,
    prompt: str,
    runtime_context: Optional[Dict[str, Any]] = None
) -> dict:
    """Create a new process from a linked program.

    Args:
        program_name: Name of the linked program to call
        prompt: The prompt to send
        runtime_context: Injected runtime context

    Returns:
        Response from the child process
    """
    # Access the process instance from runtime context
    parent_process = runtime_context["process"]

    # Implementation...
    return {"response": "Child process response"}
```

The runtime context typically contains:
- `process`: The LLMProcess instance
- `fd_manager`: File descriptor manager (if enabled)
- `linked_programs`: Dictionary of linked programs (if available)

## Registering Tools

### In the Constructor

You can register tools when creating an LLMProgram:

```python
program = LLMProgram(
    model_name="claude-3-7-sonnet",
    provider="anthropic",
    tools=[
        add_numbers,        # Function-based tool
        "read_file",        # Built-in tool by name
        weather_service.get_weather  # Instance method
    ]
)
```

### Using register_tools()

Or register them later:

```python
program.register_tools([
    add_numbers,
    "read_file",
    weather_service.get_weather
])
```

### Fluent API

Function tools work seamlessly with the fluent API:

```python
program = (
    LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant."
    )
    .register_tools([add_numbers, get_weather, fetch_data])
    .add_preload_file("context.txt")
    .add_linked_program("expert", expert_program, "A specialized expert program")
)

process = await program.start()
```

## Advanced Features

### Type Hint Support

The system automatically converts Python type hints to JSON Schema:

- Basic types: `str`, `int`, `float`, `bool`
- Complex types: `List[T]`, `Dict[K, V]`
- Optional types: `Optional[T]` (equivalent to `Union[T, None]`)
- Default values: Parameters with default values are marked as optional

### Docstring Parsing

Parameter descriptions are extracted from Google-style docstrings:

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

### Error Handling

Tool errors are automatically handled with standardized formatting:

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

If the LLM calls this with `y=0`, it will receive: `Tool 'division_tool' error: division by zero`

### Tool Metadata

You can access tool metadata programmatically:

```python
from llmproc.common.metadata import get_tool_meta

# Get metadata for a tool
meta = get_tool_meta(get_weather)
print(f"Tool name: {meta.name}")
print(f"Tool access level: {meta.access.value}")
print(f"Requires context: {meta.requires_context}")
```

## Best Practices

1. **Use Type Hints**: Always include type hints for parameters and return values
2. **Write Clear Docstrings**: Use Google-style docstrings with Args and Returns sections
3. **Explicit Parameter Descriptions**: For complex parameters, use `param_descriptions` in the decorator
4. **Handle Errors Gracefully**: Catch and handle exceptions in your tool functions
5. **Keep Tools Focused**: Design tools to do one thing well rather than multiple operations
