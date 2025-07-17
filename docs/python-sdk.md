# Python SDK

LLMProc provides a fluent, Pythonic SDK interface for creating and configuring LLM programs. This guide describes how to use the Python SDK features.

The SDK follows the Unix-inspired runtime model where programs (configuration) are distinct from processes (runtime instances). For the rationale behind this design, see the [API Design FAQ](../FAQ.md).

## Fluent API

The fluent API allows for method chaining to create and configure LLM programs:

```python
from llmproc import LLMProgram
from llmproc.plugins.preload_files import PreloadFilesPlugin

program = (
    LLMProgram(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a helpful assistant.",
        parameters={"max_tokens": 1024}  # Required parameter
    )
    .register_tools([calculator, my_tool_function])  # Enable tools using function references
    .add_plugins(PreloadFilesPlugin(["context.txt"]))
    .add_linked_program("expert", expert_program, "An expert program")
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
    .add_plugins(
        PreloadFilesPlugin(["file1.md", "file2.md"]),
        EnvInfoPlugin(EnvInfoConfig(variables=["working_directory", "platform", "date"])),
        FileDescriptorPlugin(FileDescriptorPluginConfig(max_direct_output_chars=10000)),
    )
    .register_tools([calculator, "read_file", another_function])  # Accepts both strings and callables
    .configure_thinking(budget_tokens=8192)
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
    .add_linked_program("math", math_program, "Expert in mathematics")
    .add_linked_program("code", code_program, "Expert in coding")
)
```

### Process Initialization

Programs are initialized and converted to processes with a single call to `start()`:

```python
# Start the process - this handles all initialization automatically
process = await program.start()
```

The `start()` method automatically:
- Validates the program configuration
- Loads necessary files from the program configuration
- Initializes tools and dependencies
- Creates a fully configured process instance
- Raises errors/warnings if there are any issues

This is the preferred way to create a process from a program definition.

## Creating Programs from Dictionaries

You can create programs directly from Python dictionaries without configuration files:

```python
from llmproc import LLMProgram

# Create from dictionary
config = {
    "model": {
        "name": "claude-3-5-sonnet-20241022",
        "provider": "anthropic"
    },
    "parameters": {
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "tools": {
        "builtin": ["read_file", "calculator"]
    }
}

program = LLMProgram.from_dict(config)
process = await program.start()
```

This is useful for:
- Dynamic configuration generation
- Testing with different configurations
- Integrating with configuration management systems

## Advanced Configuration

### Environment Information

Configure which environment variables are included in the system prompt by registering
the :class:`EnvInfoPlugin`:

```python
from llmproc.plugins import EnvInfoPlugin
from llmproc.config import EnvInfoConfig

program.add_plugins(
    EnvInfoPlugin(EnvInfoConfig(variables=["working_directory", "platform", "date"]))
)
```


### File Descriptor System

Configure the file descriptor system for handling large outputs by registering
the :class:`FileDescriptorPlugin`:

```python
from llmproc.plugins import FileDescriptorPlugin
from llmproc.config.schema import FileDescriptorPluginConfig

# Enable with default settings
program.add_plugins(FileDescriptorPlugin(FileDescriptorPluginConfig()))

# Configure with custom settings
program.add_plugins(
    FileDescriptorPlugin(
        FileDescriptorPluginConfig(
            max_direct_output_chars=10000,
            default_page_size=5000,
            enable_references=True,
        )
    )
)
```

### Claude 3.7 Thinking Models

Configure the thinking capability for Claude 3.7 models:

```python
# Enable thinking with default budget
program.configure_thinking()

# Enable thinking with custom budget
program.configure_thinking(budget_tokens=8192)

# Disable thinking
program.configure_thinking(enabled=False)
```

### Token-Efficient Tools

Enable token-efficient tool use for Claude 3.7 models:

```python
# Enable token-efficient tools
program.enable_token_efficient_tools()
```

### MCP Tools

The Model Context Protocol (MCP) allows integration with external tool servers. Using MCP requires two steps:

1. Configure the MCP server connection path
2. Register specific MCP tools for use

#### Configuring MCP Server

First, configure the MCP server connection:

```python
# Set up MCP server configuration from a JSON file
program.configure_mcp(config_path="config/mcp_servers.json")

# Or embed server definitions directly
program.configure_mcp(servers={"calc": {"type": "stdio", "command": "echo"}})
```

In your TOML configuration files, MCP server configuration is defined in the `[mcp]` section, and MCP tools are defined in the `[tools.mcp]` section:

```yaml
mcp:
  config_path: "config/mcp_servers.json"  # or use inline servers
  #servers:
  #  calc:
  #    type: stdio
  #    command: echo
  #    args: ["calc"]

# MCP tools configuration
tools:
  mcp:
    sequential-thinking: "all"  # Use all tools from this server
    github:
      - search_repositories
      - get_file_contents  # Specific tools
    # Example with description override
    # - name: search_repositories
    #   access: read
    #   description: Search GitHub
```

#### Registering MCP Tools

After setting up the server configuration, register MCP tools using the
`MCPServerTools` class:

```python
from llmproc.tools.mcp import MCPServerTools

# Register MCP tools using the MCPServerTools class
program.register_tools([
    # Include all tools from the "calc" server
    MCPServerTools(server="calc"),
    # Include specific tools from the "github" server
    MCPServerTools(server="github", names=["search_repositories", "get_file_contents"]),
    # Include a list of tools from the "weather" server
    MCPServerTools(server="weather", names=["current", "forecast"]),
    # Include a single tool from the "code" server with READ access
    MCPServerTools(server="code", names="explain", access="read")
])
```

#### All-in-One Initialization

You can also set up everything at once in the constructor:

```python
from llmproc.tools.mcp import MCPServerTools
from llmproc.tools.builtin import calculator, read_file

# Create program with MCP configuration and tools
program = LLMProgram(
    model_name="claude-3-7-sonnet",
    provider="anthropic",
    system_prompt="You are a helpful assistant.",
    # Configure MCP server
    mcp_config_path="config/mcp_servers.json",
    # Mix MCP tools with other tool types
    tools=[
        # MCP tools
        MCPServerTools(server="calc"),
        MCPServerTools(server="github", names="search_repositories"),
        # Built-in tools
        calculator,
        read_file
    ]
)
```

This approach provides a clean separation between server configuration and tool registration, while maintaining consistency with how other tools are registered.

## Tool Management

### Initialization in Constructor

You can pass tools directly in the LLMProgram constructor:

```python
# Using the direct constructor approach
from llmproc.tools.builtin import calculator, read_file

program = LLMProgram(
    model_name="claude-3-5-sonnet",
    provider="anthropic",
    system_prompt="You are a helpful assistant.",
    tools=[calculator, read_file, my_custom_tool]  # List of functions or string names
)
```

The tools parameter accepts:
- Callable functions (built-in tools or your own custom functions)
- String names of built-in tools (e.g., "calculator")
- A mixed list of both callable functions and string names

### Function-Based Tools

LLMProc supports registering Python functions as tools with automatic schema generation from type hints and docstrings. This allows you to easily integrate custom Python functionality with your LLM programs.

For detailed documentation on function-based tools, including:
- Basic usage and examples
- The `register_tool` decorator
- Type conversion from Python types to JSON schema
- Support for both synchronous and asynchronous functions
- Parameter validation and error handling

See the dedicated [Function-Based Tools](function-based-tools.md) documentation.

A complete working example is available in [examples/scripts/python-sdk.py](../examples/scripts/python-sdk.py) that demonstrates all SDK features including function tools, tool aliases, and advanced configuration.

### Setting and Getting Registered Tools

You can configure which tools are registered after program creation:

```python
# Setting registered tools
from llmproc.tools.builtin import calculator, read_file

# register_tools accepts both string names and callable functions
program.register_tools([calculator, "read_file", my_custom_tool])

# Get a list of all registered tool names
registered_tools = program.registered_tools
print(f"Currently registered tools: {registered_tools}")
```

The `registered_tools` property returns the string names of all registered tools.

---
[← Back to Documentation Index](index.md)
