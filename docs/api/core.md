# Core API Reference

This document serves as the canonical reference for the core API structure of the LLMProc library. Any modifications to the library should maintain these class relationships and interfaces.

## Class Hierarchy

```
LLMProgram
├── from_toml()     # Load and compile from TOML
├── compile()       # Advanced compilation options
└── start()         # Create and initialize a process

LLMProcess
├── run()           # Run with user input and callbacks
├── get_last_message() # Get response text
├── reset_state()   # Reset conversation state
├── call_tool()     # Call a tool by name
└── get_state()     # Get full conversation state

RunResult
├── api_call_infos  # Raw API response data
├── api_calls       # Count of API calls
├── duration_ms     # Duration in milliseconds
└── complete()      # Complete and calculate timing

ToolRegistry
├── register_tool() # Register a tool
├── get_handler()   # Get a tool handler
├── call_tool()     # Call a tool
└── get_definitions() # Get tool schemas
```

## Key Interfaces and Patterns

### Standard Usage Pattern

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # 1. Load and compile
    program = LLMProgram.from_toml("path/to/config.toml")
    
    # 2. Start the process
    process = await program.start()
    
    # 3. Run and get metrics
    run_result = await process.run("User input")
    
    # 4. Get the response text
    response = process.get_last_message()

asyncio.run(main())
```

### Callback Interface

```python
callbacks = {
    "on_tool_start": lambda tool_name, args: None,
    "on_tool_end": lambda tool_name, result: None,
    "on_response": lambda content: None
}

run_result = await process.run("User input", callbacks=callbacks)
```

### Tool Registration Interface

```python
# Register a tool with the registry
tool_definition = {
    "name": "example_tool",
    "description": "An example tool",
    "input_schema": {
        "type": "object",
        "properties": {"param": {"type": "string"}}
    }
}

# Define an async handler function
async def tool_handler(args: dict) -> Any:
    return {"result": f"Processed {args.get('param')}"}

# Register the tool
registry.register_tool("example_tool", tool_handler, tool_definition)
```

## Key Classes and Responsibilities

### LLMProgram

Responsible for:
- Configuration validation
- File path resolution
- Program compilation
- Creating the process

### LLMProcess

Responsible for:
- Conversation state management
- Tool execution
- API interaction
- Response handling

### RunResult

Responsible for:
- Tracking metrics
- Recording API information
- Calculating timing data

### ToolRegistry

Responsible for:
- Tool registration
- Tool access
- Tool execution

## Provider Architecture

The provider system follows this pattern:

```
providers/
├── __init__.py                    # Public interface
├── providers.py                   # Base provider classes
├── anthropic_process_executor.py  # Anthropic-specific executor
└── anthropic_tools.py             # Anthropic tool handling
```

Each provider should implement:
1. A client factory function
2. A process executor class
3. Tool integration functions

## Tools Architecture

The tools system follows this pattern:

```
tools/
├── __init__.py      # Tool registry and public interface
├── spawn.py         # Spawn tool implementation
├── fork.py          # Fork tool implementation
└── mcp.py           # MCP tools integration
```

Each tool should:
1. Define its schema
2. Implement its handler function
3. Register with the ToolRegistry

## Guidelines for Modifications

1. **Maintain Async Pattern**: All potentially long-running operations should be async.
2. **Use RunResult**: All run operations should return RunResult objects.
3. **Keep Callbacks Optional**: All callback parameters should be optional.
4. **Clear Separation**: Maintain separation between program definition and process execution.
5. **Sensible Defaults**: Provide reasonable defaults for all optional parameters.
6. **Proper Error Handling**: Ensure all async operations have proper error handling.
7. **Type Annotations**: Use proper type annotations for all public methods.
8. **Documentation**: Update this reference when changing core interfaces.