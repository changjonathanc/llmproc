# Python SDK

LLMProc provides a fluent, Pythonic SDK interface for creating and configuring LLM programs. This guide describes how to use the Python SDK features implemented in [RFC018](../RFC/RFC018_python_sdk.md).

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
    .add_preload_file("context.txt")
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
    .add_preload_file("file1.md")
    .add_preload_file("file2.md")
    .add_tool(tool_function)
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

### Compilation

All programs are compiled before starting:

```python
# Compile the program
program.compile()

# Start the process
process = await program.start()
```

compile() will load necessary files from the program configuration and raise error/warning if there's any issue. It will be called automatically when start() is called if the program is not compiled.

So you can call start() directly.

```python
process = await program.start()
```

## Function-Based Tools

LLMProc supports registering Python functions as tools with automatic schema generation from type hints and docstrings. This allows you to easily integrate custom Python functionality with your LLM programs.

For detailed documentation on function-based tools, including:
- Basic usage and examples
- The `register_tool` decorator
- Type conversion from Python types to JSON schema
- Support for both synchronous and asynchronous functions
- Parameter validation and error handling

See the dedicated [Function-Based Tools](function-based-tools.md) documentation.

A complete working example is also available in [examples/features/function_tools.py](../examples/features/function_tools.py).