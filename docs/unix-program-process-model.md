# Unix-Inspired Program/Process Model

LLMProc implements a Unix-inspired runtime approach to separating program configuration from process execution. This document explains the design principles, architecture, and usage of this model.

## Core Concepts

The Unix model provides a clear distinction between:

1. **Program**: Static definition and configuration (similar to an executable in Unix)
2. **Process**: Runtime execution and state (similar to a running process in Unix)

This separation offers several advantages:
- **Cleaner Architecture**: Clear responsibilities for each component
- **Improved Testability**: Components can be tested in isolation
- **Reduced Dependencies**: Eliminates circular dependencies
- **Better Resource Management**: Clear lifecycle for resources

For the design rationale behind this architecture, see the [API Design FAQ](../FAQ.md#why-separate-program-and-process-in-the-api).

## Architecture Overview

### Unidirectional Flow

The main architectural principle is a unidirectional flow:

```
Program → Tool Definitions → Process → Runtime Context → Tool Execution
```

This replaces the previous circular dependency:

```
Program → Process → ToolManager → Process (circular dependency)
```

### Key Components

1. **LLMProgram**: Handles configuration and compilation
   - Loads TOML configuration
   - Validates settings
   - Defines tools and their schemas
   - Contains static definitions

2. **LLMProcess**: Manages runtime execution
   - Contains conversation state
   - Coordinates API calls
   - Manages file descriptors
   - Provides runtime context for tools

3. **Runtime Context**: Dependency injection container
   - Contains references to runtime components (process, file descriptors, etc.)
   - Injected into tools during execution
   - Decouples tools from process implementation

4. **Context-Aware Tools**: Tools that explicitly declare their runtime dependencies
   - Defined with `register_tool(requires_context=True)`
   - Receive runtime context during execution
   - Extract only the dependencies they need

## Implementation Details

### Context-Aware Tool Registration

The `register_tool` decorator with `requires_context=True` marks tools that require runtime access:

```python
from llmproc.tools.function_tools import register_tool

@register_tool(
    requires_context=True,
    required_context_keys=["process", "fd_manager"]
)
async def my_tool(param: str, runtime_context=None):
    # Extract dependencies - validation happens automatically
    process = runtime_context["process"]
    fd_manager = runtime_context["fd_manager"]

    # Use dependencies
    # ...

    return result
```

This explicitly marks tools that need runtime dependencies, specifies which context keys are required, and automatically validates that they're present.

### Runtime Context Contents

The runtime context typically contains:

| Key | Value | Description |
|-----|-------|-------------|
| `process` | LLMProcess instance | The running process instance |
| `fd_manager` | FileDescriptorManager | File descriptor management system |
| `linked_programs` | dict | Available linked programs |
| `linked_program_descriptions` | dict | Descriptions of linked programs |

Tools only extract the dependencies they need, creating a cleaner interface.

### Tool Initialization with Configuration

Tool initialization uses a configuration-based approach to avoid circular dependencies:

```python
# Create a configuration dictionary
config = {
    "fd_manager": fd_manager,
    "linked_programs": program.linked_programs,
    "linked_program_descriptions": program.linked_program_descriptions,
    "has_linked_programs": bool(program.linked_programs),
    "provider": program.provider,
    "mcp_enabled": program.mcp_config_path is not None,
    "mcp_config_path": program.mcp_config_path,
    "mcp_tools": program.mcp_tools
}

# Initialize tools with configuration
await tool_manager.initialize_tools(config)
```

The configuration dictionary contains all dependencies tools need at initialization time, without containing a direct reference to the process instance. This breaks the circular dependency where tools required process, but process required tools.

Key components in the configuration:

1. **Resource References**: References to resources like `fd_manager` and `linked_programs`
2. **Capability Flags**: Flags like `has_linked_programs` and `mcp_enabled`
3. **Configuration Paths**: Paths like `mcp_config_path` for external resources

This approach separates tool registration (which doesn't need process) from tool execution (which uses runtime context).

### Process Initialization Flow

The proper initialization flow follows these steps:

1. **Program Loading**: Load configuration from TOML
2. **Tool Definition**: Define tool schemas and handlers
3. **Process Creation**: Create process instance with program
4. **Context Setup**: Set up runtime context with dependencies
5. **Tool Registration**: Register tools with runtime registry
6. **Execution**: Run the process with proper dependency injection

This is simplified with the `program.start()` method, which handles steps 3-5 automatically.

## Best Practices

### Creating New Tools

When creating new tools:

1. Use `register_tool` with `requires_context=True` for tools that need runtime access
2. Specify which context keys are required via `required_context_keys` parameter
3. Validation happens automatically - directly access context keys that were declared
4. Return a ToolResult object with proper success/error handling

Example:

```python
from llmproc.tools.function_tools import register_tool
from llmproc.common.results import ToolResult

@register_tool(
    requires_context=True,
    required_context_keys=["fd_manager"]
)
async def my_tool(param1: str, param2: int = 0, runtime_context=None) -> ToolResult:
    """A tool that requires access to the file descriptor manager.

    Args:
        param1: First parameter description
        param2: Second parameter description
        runtime_context: Runtime context with fd_manager (validated automatically)

    Returns:
        ToolResult with the operation result
    """
    # The decorator already validated fd_manager is present
    # So we can safely access it directly
    fd_manager = runtime_context["fd_manager"]

    try:
        # Tool implementation
        result = f"Processed {param1} with value {param2}"
        return ToolResult.from_success(result)
    except Exception as e:
        return ToolResult.from_error(str(e))
```

### Initializing Processes

Always use the proper initialization flow:

```python
# Program configuration
program = LLMProgram.from_toml("config.toml")

# Process creation with proper initialization
process = await program.start()

# Now the process is fully initialized with proper runtime context
```

## Advanced Usage

### Custom Runtime Context

You can add custom values to the runtime context:

```python
# Get the program
program = LLMProgram.from_toml("config.toml")

# Create the process
process = await program.start()

# Add custom values to runtime context
process.tool_manager.set_runtime_context({
    **process.tool_manager.runtime_context,  # Preserve existing values
    "custom_data": my_custom_data,
    "custom_service": my_service
})
```

Tools can then access these values:

```python
@register_tool(
    requires_context=True,
    required_context_keys=["custom_data", "custom_service"]
)
async def my_tool(args, runtime_context=None):
    custom_data = runtime_context["custom_data"]
    custom_service = runtime_context["custom_service"]
    # Use custom dependencies
```

### Testing Tools in Isolation

The runtime context pattern simplifies testing:

```python
import pytest

@pytest.mark.asyncio
async def test_my_tool():
    # Create mock dependencies
    mock_fd_manager = MockFileDescriptorManager()

    # Create mock runtime context
    mock_context = {
        "fd_manager": mock_fd_manager
    }

    # Call tool with mock context
    result = await my_tool({"param1": "test"}, runtime_context=mock_context)

    # Verify result
    assert result.is_error is False
    assert "Processed test" in result.content
```

## Anti-Patterns to Avoid

When working with the Unix-inspired model, avoid these common anti-patterns:

### 1. Direct LLMProcess Construction

```python
# ANTI-PATTERN: Direct construction bypasses proper initialization
process = LLMProcess(program=program)  # WRONG!

# CORRECT PATTERN: Use the start() method
process = await program.start()  # Correct!
```

Direct construction bypasses the proper tool initialization flow and will result in tools that don't have access to runtime context.

### 2. Passing Process to Tool Registration

```python
# ANTI-PATTERN: Passing process to tool methods
tool_manager.register_system_tools(process)  # WRONG!

# CORRECT PATTERN: Use configuration-based approach
config = program.get_tool_configuration()
tool_manager.register_system_tools(config)  # Correct!
```

Tools should be initialized with configuration dictionaries, not direct process references.

### 3. Direct Process Access in Tool Handlers

```python
# ANTI-PATTERN: Direct process parameter
async def tool_handler(args, process):  # WRONG!
    # ...

# CORRECT PATTERN: Use register_tool with requires_context
@register_tool(requires_context=True)
async def tool_handler(args, runtime_context=None):  # Correct!
    process = runtime_context.get("process")
    # ...
```

Tool handlers should use the context-aware pattern to access runtime dependencies.

## Migration Guide

If you're updating existing tools to use the new pattern:

1. Use `@register_tool(requires_context=True)` for tool functions that need runtime access
2. Change parameter from `llm_process` to `runtime_context=None`
3. Update implementation to extract dependencies from the context
4. Update any tool registration to use the configuration-based approach
5. Update tests to provide a mock runtime context instead of process instances

Before:
```python
async def tool_handler(args, llm_process):
    # Direct process access
    result = process_data(llm_process, args)
    return result
```

After:
```python
from llmproc.tools.function_tools import register_tool

@register_tool(requires_context=True)
async def tool_handler(args, runtime_context=None):
    # Get process from context
    process = runtime_context.get("process")

    # Use process
    result = process_data(process, args)
    return result
```

## Conclusion

The Unix-inspired program/process model creates a cleaner architecture with clear separation of concerns, making the codebase more maintainable, testable, and extensible. By explicitly marking runtime dependencies with the context-aware pattern, we eliminate circular dependencies and improve the overall design.
