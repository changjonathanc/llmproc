# Core API Reference

This document serves as the canonical reference for the core API structure of the LLMProc library. Any modifications to the library should maintain these class relationships and interfaces.

## Class Hierarchy

```
LLMProgram
├── from_toml()     # Load program from TOML
├── from_yaml()     # Load program from YAML
├── start()         # Create and initialize a process (handles validation automatically)
├── register_tools() # Configure tools (accepts strings and/or callables)
├── set_tool_aliases() # Set LLM-friendly aliases for tools
├── set_user_prompt() # Set initial user prompt to execute automatically
├── set_max_iterations() # Set maximum iterations for tool calls
└── tool_manager    # Central manager for all tools

LLMProcess
├── run()           # Run with user input and callbacks
├── get_last_message() # Get response text
├── reset_state()   # Reset conversation state (⚠️ Experimental API)
├── call_tool()     # Call a tool by name
├── count_tokens()  # Count tokens in current conversation
└── get_state()     # Get full conversation state

RunResult
├── api_call_infos  # Raw API response data
├── api_calls       # Count of API calls
├── duration_ms     # Duration in milliseconds
├── tool_calls      # List of tool calls made
└── complete()      # Complete and calculate timing

ToolManager
├── initialize_tools() # Set up all tools from configuration
├── call_tool()     # Call a tool by name
├── get_tool_schemas() # Get schemas for enabled tools
└── register_tools() # Configure which tools are available

ToolRegistry
├── register_tool() # Register a tool
├── get_handler()   # Get a tool handler
├── call_tool()     # Call a tool
└── get_definitions() # Get tool schemas
```

## Key Interfaces and Patterns

### Standard Usage Patterns

#### Interactive Pattern

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # 1. Load program configuration
    program = LLMProgram.from_file("path/to/config.yaml")  # or .toml

    # 2. Start the process (handles validation automatically)
    process = await program.start()

    # 3. Run and get metrics
    run_result = await process.run("User input")

    # 4. Get the response text
    response = process.get_last_message()

asyncio.run(main())
```

#### Automatic User Prompt Pattern

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # 1. Load program configuration with user prompt
    program = LLMProgram.from_file("path/to/config.yaml")  # or .toml

    # Optional: set or override user prompt programmatically
    program.set_user_prompt("What are the key features of LLMProc?")
    program.set_max_iterations(15)  # Override default max_iterations

    # 2. Start the process (handles validation automatically)
    # If user_prompt is set, it will be executed automatically
    process = await program.start()

    # 3. No need to call process.run() unless you want to run additional prompts
    # The result of the automatic execution is available in the process
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

## Key Classes and Dependencies

The library follows a carefully designed dependency structure to minimize circular dependencies and promote separation of concerns. Here's how the key classes interact:

### LLMProgram

**Responsibilities:**
- Configuration validation
- File path resolution
- Program compilation
- Creating the process
- Providing tool configuration
- Managing user prompt configuration
- Setting maximum iterations for tool calls
- Supporting demo mode configuration

**Dependencies:**
- Uses **ProgramLoader** for configuration loading (TOML or YAML)
- Creates **ToolManager** for tool definition and registration
- Creates **LLMProcess** instances through `start()` method

### LLMProcess

**Responsibilities:**
- Conversation state management
- API interaction and provider communication
- Response handling and processing
- Runtime context management
- File descriptor management (when enabled)
- Automatic execution of user prompts
- Enforcing maximum iterations for tool calls

**Dependencies:**
- Depends on **LLMProgram** for configuration
- Uses provider-specific **ProcessExecutor** for API calls
- Contains **ToolManager** for tool access
- Contains **FileDescriptorManager** when enabled

### RunResult

**Responsibilities:**
- Tracking metrics and timing
- Recording API information
- Calculating timing data
- Recording tool calls

**Dependencies:**
- No dependencies on other core classes
- Acts as a data container for run metrics

### ToolManager

**Responsibilities:**
- Central management of all tools
- Tool initialization and setup
- Tool access control and enablement
- Runtime context injection for tools requiring runtime context
- Tool alias resolution and mapping

**Dependencies:**
- Contains two **ToolRegistry** instances:
  - `runtime_registry`: For active tool execution
  - `mcp_registry`: For MCP tool definitions
- May contain **MCPManager** for MCP tool registration

### ToolRegistry

**Responsibilities:**
- Tool registration and storage
- Tool handler access and retrieval
- Direct tool execution
- Tool schema management
- Alias resolution for registered tools

**Dependencies:**
- No dependencies on other core classes
- Acts as a self-contained registry for tools

### Class Interaction Flow

The interaction between these components follows a unidirectional flow:

1. **Configuration Phase**:
   ```
   Config File (TOML/YAML) → LLMProgram → ToolManager (Definition) → Tool Schemas
   ```

2. **Initialization Phase**:
   ```
   LLMProgram.start() → LLMProcess → ToolManager (Registration) → Runtime Context
   ```

3. **Execution Phase**:
   ```
   User Input → LLMProcess.run() → ProcessExecutor → LLM API → Tool Calls → RunResult
   ```

4. **Tool Execution Flow**:
   ```
   LLMProcess.call_tool() → ToolManager.call_tool() → Tool Handler (with Runtime Context)
   ```

This architecture ensures that:
- Configuration and runtime concerns are separated
- Tools can be defined without circular dependencies
- Runtime dependencies are explicitly injected only where needed
- Processes interact with tools through a controlled interface

## Provider Architecture

The provider system follows this pattern:

```
providers/
├── __init__.py                     # Public interface
├── providers.py                    # Client factory functions
├── utils.py                        # Shared utility functions
├── constants.py                    # Provider constants and lists
├── anthropic_process_executor.py   # Anthropic-specific executor
├── anthropic_utils.py              # Anthropic-specific utilities
├── openai_process_executor.py      # OpenAI-specific executor
└── gemini_process_executor.py      # Gemini-specific executor
```

Each provider should implement:
1. A client factory function in providers.py
2. A process executor class with run() method
3. Token counting functionality
4. Tool handling for provider-specific formats

## Tools Architecture

The tools system follows this pattern:

```
tools/
├── __init__.py           # Public interface
├── tool_manager.py       # Central tool management
├── tool_registry.py      # Tool registration and access
├── context_aware.py      # Context-aware tool decorator
├── builtin/
│   ├── __init__.py       # Builtin tools public interface
│   ├── spawn.py          # Spawn tool implementation
│   ├── fork.py           # Fork tool implementation
│   ├── goto.py           # Goto tool implementation
│   └── fd_tools.py       # File descriptor tools
└── mcp/
    ├── __init__.py       # MCP public interface
    ├── manager.py        # MCP server manager
    └── integration.py    # MCP tools integration
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
