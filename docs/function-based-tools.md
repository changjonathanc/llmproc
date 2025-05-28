# Function-Based Tools

LLMProc supports registering Python functions as tools with automatic schema generation from type hints and docstrings. This provides a simple and intuitive way to create tools without writing boilerplate tool definition code.

## Basic Usage

```python
from llmproc import LLMProgram, register_tool
from llmproc.tools.builtin import calculator  # Import built-in tool functions

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
        system_prompt="You are a helpful assistant.",
        # Tools can be provided directly in the constructor
        tools=[calculator, get_calculator]  # List of callable functions
    )
)

# Or set tools after creation
# program.register_tools([calculator, get_calculator, "read_file"])

# Start the LLM process
process = await program.start()
```

## Optional: Using the `register_tool` Decorator

The `register_tool` decorator is entirely optional. You can register plain functions as tools without any decoration, and the system will automatically extract information from function signatures, docstrings, and type hints.

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

The `param_descriptions` argument allows you to explicitly define parameter descriptions instead of relying on docstring parsing, which should be considered a fallback mechanism. Explicit parameter descriptions provide more control and clarity in your tool schemas. When you override a single parameter using `ToolConfig` or `MCPServerTools`, LLMProc merges your override with the existing descriptions so that unspecified parameters retain their built-in text.

## Class Instance Methods as Tools

LLMProc supports registering class instance methods as tools, allowing for stateful tools where the method has access to the instance state:

```python
from typing import Dict, Any
from llmproc import LLMProgram, register_tool

class DataProvider:
    def __init__(self):
        self.counter = 0
        self.data = {"users": ["Alice", "Bob"]}

    def get_data(self, key: str) -> Dict[str, Any]:
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

    async def fetch_remote(self, resource_id: str) -> Dict[str, Any]:
        """Fetch data from remote resource.

        Args:
            resource_id: ID of the resource to fetch

        Returns:
            The fetched resource data
        """
        # Async implementation
        self.counter += 1
        return {"id": resource_id, "access_count": self.counter}

# Create an instance
provider = DataProvider()

# Method 1: Register instance methods directly
program = LLMProgram(
    model_name="claude-3-7-sonnet",
    provider="anthropic"
)
program.register_tools([
    provider.get_data,      # Registers the bound method
    provider.fetch_remote   # Works with async methods too
])

# Method 2: Apply register_tool decorator to instance methods
# This enables full customization of the tool metadata
decorated_method = register_tool(
    name="get_provider_data",
    description="Get data from the provider with access tracking"
)(provider.get_data)

program.register_tools([decorated_method])
```

When registering instance methods as tools:

1. The method maintains access to the instance state (`self`) when called
2. Both sync and async instance methods are supported
3. The `self` parameter is automatically handled and not exposed to the LLM
4. You can apply `register_tool` decorator to instance methods for advanced customization

> **Implementation Note**: When using `register_tool` on an instance method, internally a wrapper function is created to hold the metadata. This wrapper is for internal use by llmproc only and should not be called directly in your code. Always interact with the returned decorated method through the LLMProcess API.

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
        system_prompt="You are a helpful assistant.",
        parameters={"max_tokens": 1024}  # Required parameter
    )
    .register_tools([get_calculator, get_weather, fetch_data])  # Enable multiple tools at once
    .add_preload_file("context.txt")
    .add_linked_program("expert", expert_program, "A specialized expert program")
)

# Create process from the program
process = await program.start()
```

## Mixed Tool Types

You can mix function-based tools with string-based tool names in both the constructor and register_tools():

```python
# Enable both function-based and built-in tools
program = (
    LLMProgram(
        model_name="claude-3-5-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant.",
        # Mix tools in constructor
        tools=[
            get_calculator,         # Function-based tool
            "read_file",            # Built-in tool by name
            "calculator"            # Another built-in tool
        ]
    )
)

# Or mix tools when calling register_tools()
program.register_tools([
    get_calculator,         # Function-based tool
    "read_file",            # Built-in tool by name
    "calculator"            # Another built-in tool
])
```

## Tool Error Handling

Tool errors are automatically handled and returned as proper error responses with standardized formatting:

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

When the LLM tries to call this tool with `y=0`, it will receive a proper error message with the standardized format `Tool '{name}' error: {message}`. For example: `Tool 'division_tool' error: division by zero`.

All errors from function tools follow this consistent pattern, making error handling and debugging easier.

## Tool Metadata

Function tools use a central metadata system that stores all tool-related information:

```python
from llmproc.common.access_control import AccessLevel
from llmproc.common.metadata import ToolMeta, get_tool_meta

# Decorated tools automatically get their metadata configured
@register_tool(
    name="weather_info",
    description="Get weather information",
    access=AccessLevel.READ
)
def get_weather(location: str):
    # implementation...
    pass

# You can access the metadata after decoration
meta = get_tool_meta(get_weather)
print(f"Tool name: {meta.name}")
print(f"Tool access level: {meta.access.value}")
print(f"Requires context: {meta.requires_context}")
```

The metadata system stores all tool properties in a single location, maintaining a clean interface and avoiding attribute pollution.

## Context-Aware Tools

You can create tools that require access to the LLMProcess runtime context using the `requires_context=True` parameter:

```python
from typing import Optional, Dict, Any
from llmproc import register_tool

@register_tool(
    name="spawn_tool",
    description="Create a new process from a linked program",
    param_descriptions={
        "program_name": "Name of the linked program to call",
        "prompt": "The prompt to send to the linked program"
    },
    requires_context=True,  # Mark tool as requiring runtime context
    required_context_keys=["process"]  # Specify required context keys
)
async def spawn_child_process(
    program_name: str,
    prompt: str,
    runtime_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
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

When a tool is registered with `requires_context=True`, the system:
1. Stores this information in the tool's metadata
2. Automatically validates context requirements at runtime
3. Injects the runtime_context parameter when the tool is called

The runtime context typically contains:
- `process`: The LLMProcess instance
- `fd_manager`: File descriptor manager (if enabled)
- `linked_programs`: Dictionary of linked programs (if available)

## Initialization

Function tools are processed during process initialization. When you call `program.start()`:

1. Extracts schema information from type hints and docstrings
2. Creates async-compatible tool handlers
3. Registers tools with the tool registry
4. Sets up runtime context injection for context-aware tools

When the process is started, the tools are ready to use by the LLM.
