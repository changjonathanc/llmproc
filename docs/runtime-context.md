# Runtime Context Management

Runtime context is a core part of the LLMProc architecture that enables dependency injection for tools without creating circular dependencies. This document explains how runtime context works and how to use it effectively.

## Overview

The runtime context system provides a standardized way for tools to:

1. Access runtime components like LLMProcess, FileDescriptorManager, etc.
2. Explicitly declare their dependencies
3. Validate required components at runtime
4. Handle missing dependencies gracefully

## Using Context-Aware Tools

### Defining a Context-Aware Tool

The recommended way to create a context-aware tool is to use the `@register_tool` decorator with `requires_context=True`. This decorator handles validation automatically:

```python
from llmproc.tools.function_tools import register_tool

@register_tool(
    requires_context=True
)
async def example_tool(param1: str, param2: int, runtime_context=None):
    # Access the process from the context
    process = runtime_context["process"]

    # Use the process to do something
    return f"Process model: {process.model_name}"
```

### Specifying Required Context Keys

For better reliability, always specify which context keys your tool requires using the `required_context_keys` parameter:

```python
@register_tool(
    requires_context=True,
    required_context_keys=["process", "linked_programs"]
)
async def spawn_tool(program_name: str, prompt: str, runtime_context=None):
    # Context validation happens automatically via the decorator
    # You can safely access the required keys without additional checks
    process = runtime_context["process"]
    linked_programs = runtime_context["linked_programs"]

    # Implementation...
    return result
```

### How Validation Works

When you use `@register_tool` with `requires_context=True` and `required_context_keys`, the decorator:

1. Wraps your function with a validator that checks the runtime_context
2. Automatically verifies all specified keys are present
3. Returns a proper error ToolResult if validation fails
4. Only calls your function if all required context is available

This means you don't need to add manual validation in your tool function!

## Runtime Context Structure

The `RuntimeContext` is defined as a TypedDict with the following keys:

```python
class RuntimeContext(TypedDict, total=False):
    process: Any  # LLMProcess instance
    fd_manager: Any  # FileDescriptorManager instance
    linked_programs: dict[str, Any]  # Dictionary of linked programs
    linked_program_descriptions: dict[str, str]  # Dictionary of program descriptions
    stderr: list[str]  # Buffer for logging via write_stderr tool
```

## Runtime Context Initialization

Runtime context is initialized in the `program_exec.py` module:

```python
# Example of creating and setting up runtime context
from llmproc.program_exec import setup_runtime_context

# Create context for a process
context = setup_runtime_context(process)
```

## Logging to Standard Error

The built-in `write_stderr` tool appends log messages to the
`stderr` buffer in the runtime context. Retrieve the accumulated log with
`process.get_stderr_log()`.

```python
process.call_tool("write_stderr", {"message": "starting step"})
print(process.get_stderr_log())
```

## Best Practices

1. **Use register_tool**: Always use `@register_tool(requires_context=True, required_context_keys=[...])` for context-aware tools
2. **Be Explicit**: Always specify exactly which context keys your tool requires via `required_context_keys`
3. **Skip Manual Validation**: Let the decorator handle validation - don't add redundant validation code
4. **Return ToolResult**: Always return a ToolResult object from your tool functions for consistent error handling

## Debugging Context Issues

If you encounter runtime context issues:

1. Check that your tool is properly marked with `requires_context=True`
2. Verify that you've specified all required context keys via `required_context_keys`
3. Ensure the tools are being called through `process.call_tool()` or `tool_manager.call_tool()`
4. Check that the process has all required components (fd_manager, linked_programs, etc.)
5. Verify that `setup_runtime_context()` is being called during process initialization

## Implementation Details

Runtime context is set up in two main locations:

1. In `program_exec.py` during the program-to-process transition
2. In the `tool_manager.set_runtime_context()` method

The context is validated in:

1. The wrapper created by `register_tool` for tools with explicit requirements
2. The `ToolManager.call_tool()` method for all context-aware tools
