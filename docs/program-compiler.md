# Program Compiler Feature

## Overview

The Program Compiler feature provides a robust way to validate, load, and process configuration files (TOML or YAML) before instantiating an `LLMProcess`. This separation of concerns makes the codebase more maintainable and easier to extend. The compiler now uses a global registry and builds a complete object graph of compiled programs.

## Key Benefits

1. **Validation**: Automatic validation of program definitions with clear error messages using Pydantic
2. **Separation of Concerns**: Moves program parsing logic out of `LLMProcess`
3. **Reusability**: Allows programs to be compiled once and used multiple times
4. **Extensibility**: Makes it easier to add new program options in the future
5. **Memory Efficiency**: Program registry avoids redundant compilation
6. **Object Graph**: Direct references between compiled programs create a proper object graph
7. **Lazy Instantiation**: Programs are only instantiated as processes when needed
8. **Async Initialization**: Programs can be started with proper async initialization

## Global Program Registry

The Program Compiler uses a singleton registry to store compiled programs:

```python
from llmproc import LLMProgram

# First compilation of a program
program1 = LLMProgram.from_toml("path/to/program.toml")

# Second request for the same program retrieves it from the registry
program2 = LLMProgram.from_toml("path/to/program.toml")

# Both variables reference the same object
assert program1 is program2  # True
```

## API

### Using LLMProgram.from_toml

```python
from llmproc import LLMProgram

# Load and compile a program
program = LLMProgram.from_toml("path/to/program.toml")

# Access program properties
print(f"Model: {program.model_name}")
print(f"Provider: {program.provider}")
print(f"System Prompt: {program.system_prompt}")
print(f"API Parameters: {program.api_params}")
print(f"Linked Programs: {program.linked_programs}")  # Contains Program objects
```

### Using Advanced Compilation Options

```python
from llmproc import LLMProgram

# Compile without processing linked programs
standalone_program = LLMProgram.compile("path/to/program.toml", include_linked=False)

# Get a dictionary of all compiled programs in the graph
all_programs = LLMProgram.compile("path/to/program.toml", return_all=True)

# Skip checking if linked files exist (useful for validation only)
validated_program = LLMProgram.compile("path/to/program.toml", check_linked_files=False)
```

### Starting a Process from a Compiled Program

```python
import asyncio
from llmproc import LLMProgram

# Load and compile the program
program = LLMProgram.from_toml("path/to/program.toml")

# Start the process with async initialization
async def main():
    # Start the process (handles async initialization)
    process = await program.start()

    # Get metrics for the run
    run_result = await process.run("Hello, how are you?")

    # Get the assistant's response
    response = process.get_last_message()
    print(f"Response: {response}")

    # Display metrics
    print(f"API calls: {run_result.api_calls}")
    print(f"Duration: {run_result.duration_ms}ms")

# Run the async function
asyncio.run(main())
```

### Using Callbacks for Real-time Updates

```python
import asyncio
from llmproc import LLMProgram

async def main():
    program = LLMProgram.from_toml("path/to/program.toml")
    process = await program.start()

    # Define callbacks
    callbacks = {
        "on_tool_start": lambda tool_name, args: print(f"Starting tool: {tool_name}"),
        "on_tool_end": lambda tool_name, result: print(f"Tool completed: {tool_name}"),
        "on_response": lambda content: print(f"Received response: {content[:30]}...")
    }

    # Run with callbacks
    run_result = await process.run("What can you tell me about Python?", callbacks=callbacks)

    # Get the final response
    response = process.get_last_message()
    print(f"Final response: {response}")

    # Print run metrics
    print(f"Run completed in {run_result.duration_ms}ms")
    print(f"API calls: {run_result.api_calls}")

asyncio.run(main())
```

### Simplified Synchronous API

```python
from llmproc import LLMProgram

# Create a program and start the process in one session
program = LLMProgram.from_toml("path/to/program.toml")
process = program.start_sync()
```

## Validation Features

The program compiler validates:

- Required fields like model name and provider
- Provider compatibility (must be one of 'openai', 'anthropic', or 'vertex')
- Tool configuration formats for MCP
- File path existence for system prompts, preloaded files, and MCP configurations
- Proper format of parameters for each provider

## Error Handling

When validation fails, the compiler provides clear error messages with specific information about what went wrong:

```
Invalid program in path/to/program.toml:
1 validation error for LLMProgramConfig
model -> name
  field required (type=value_error.missing)
```

## Program Structure

A compiled program includes these components:

- `model_name`: Name of the model to use
- `provider`: Provider of the model ('openai', 'anthropic', or 'vertex')
- `system_prompt`: System prompt that defines the behavior of the process
- `parameters`: Dictionary of parameters for the LLM
- `api_params`: Extracted API parameters (temperature, max_tokens, etc.)
- `display_name`: User-facing name for the process
- `preload_files`: List of files to preload into the system prompt
- `mcp_config_path`: Path to MCP configuration file
- `mcp_tools`: Dictionary of MCP tools to enable
- `tools`: Dictionary of built-in tools configuration
- `linked_programs`: Dictionary of linked programs (references to Program objects)
- `base_dir`: Base directory for resolving relative paths
- `compiled`: Flag indicating whether the program is fully compiled
- `source_path`: Path to the source configuration file
- `env_info`: Environment information configuration

## RunResult Object

The new `RunResult` class provides detailed metrics about each run:

```python
class RunResult:
    """Contains metadata about a process run."""

    api_call_infos: List[Dict[str, Any]]  # Raw API response data
    api_calls: int                        # Number of API calls made
    start_time: float                     # When the run started
    end_time: Optional[float]             # When the run completed
    duration_ms: int                      # Duration in milliseconds
```

All process runs now return a `RunResult` object for detailed metrics, diagnostics, and resource usage tracking.

## Object Graph and Linked Programs

With the compilation semantics, the `linked_programs` attribute contains direct references to compiled Program objects:

```python
from llmproc import LLMProgram

# Compile a program with linked programs
main_program = LLMProgram.from_toml("path/to/main.toml")

# Access linked programs directly as Program objects
expert_program = main_program.linked_programs["expert"]
print(f"Expert model: {expert_program.model_name}")
```

## Async Initialization and Lazy Instantiation

Programs can be started asynchronously when they need initialization, and linked programs are lazily instantiated as processes only when needed:

```python
from llmproc import LLMProgram

async def main():
    # Load the program configuration
    main_program = LLMProgram.from_toml("path/to/main.toml")

    # Start the process with async initialization
    main_process = await main_program.start()

    # Linked programs are instantiated only when used via tools
    run_result = await main_process.run("Use the expert to analyze this code.")

asyncio.run(main())
```

## Implementation Details

- Uses Pydantic models for validation
- Resolves relative file paths based on the configuration file location
- Extracts API parameters for convenient access
- Provides both synchronous compilation and async initialization methods
- Uses a singleton registry to avoid redundant compilation
- Implements non-recursive BFS for program graph traversal
- Implements two-phase compilation
- Handles circular dependencies gracefully
- Provides better error messages with file path information
- Supports skipping file existence checks for validation-only scenarios
- Includes detailed metrics tracking with RunResult
- Supports callback-based event monitoring
