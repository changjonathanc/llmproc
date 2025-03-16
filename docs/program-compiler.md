# Program Compiler Feature

## Overview

The Program Compiler feature provides a robust way to validate, load, and process TOML program configurations before instantiating an `LLMProcess`. This separation of concerns makes the codebase more maintainable and easier to extend. The compiler now uses a global registry and builds a complete object graph of compiled programs.

## Key Benefits

1. **Validation**: Automatic validation of program configurations with clear error messages using Pydantic
2. **Separation of Concerns**: Moves configuration parsing logic out of `LLMProcess`
3. **Reusability**: Allows programs to be compiled once and used multiple times
4. **Extensibility**: Makes it easier to add new configuration options in the future
5. **Memory Efficiency**: Program registry avoids redundant compilation
6. **Object Graph**: Direct references between compiled programs create a proper object graph
7. **Lazy Instantiation**: Programs are only instantiated as processes when needed

## Global Program Registry

The Program Compiler now uses a singleton registry to store compiled programs:

```python
from llmproc import LLMProgram

# First compilation of a program
program1 = LLMProgram.compile("path/to/program.toml")

# Second request for the same program retrieves it from the registry
program2 = LLMProgram.compile("path/to/program.toml")

# Both variables reference the same object
assert program1 is program2  # True
```

## API

### Using LLMProgram.compile

```python
from llmproc import LLMProgram

# Basic compilation
program = LLMProgram.compile("path/to/program.toml")

# Compile without processing linked programs
standalone_program = LLMProgram.compile("path/to/program.toml", include_linked=False)

# Get a dictionary of all compiled programs in the graph
all_programs = LLMProgram.compile("path/to/program.toml", return_all=True)

# Skip checking if linked files exist (useful for validation only)
validated_program = LLMProgram.compile("path/to/program.toml", check_linked_files=False)

# Access program properties
print(f"Model: {program.model_name}")
print(f"Provider: {program.provider}")
print(f"System Prompt: {program.system_prompt}")
print(f"API Parameters: {program.api_params}")
print(f"Linked Programs: {program.linked_programs}")  # Now contains Program objects
```

### Creating an LLMProcess from a Compiled Program

```python
import llmproc
from llmproc import LLMProgram

# Compile the program
program = LLMProgram.compile("path/to/program.toml")

# Create an LLMProcess from the compiled program
process = program.instantiate(llmproc)

# Use the process
response = await process.run("Hello, how are you?")
```

### Using LLMProcess.from_toml (now uses the compiler internally)

```python
from llmproc import LLMProcess

# Create a process directly from a TOML file (uses the compiler internally)
process = LLMProcess.from_toml("path/to/program.toml")
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
Invalid program configuration in path/to/program.toml:
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
- `linked_programs`: Dictionary of linked programs (now references Program objects after compilation)
- `debug_tools`: Flag to enable detailed tool debugging output
- `base_dir`: Base directory for resolving relative paths
- `compiled`: Flag indicating whether the program is fully compiled (including linked programs)
- `source_path`: Path to the source TOML file

## Object Graph and Linked Programs

With the new compilation semantics, the `linked_programs` attribute now contains direct references to compiled Program objects instead of string paths:

```python
from llmproc import LLMProgram

# Compile a program with linked programs
main_program = LLMProgram.compile("path/to/main.toml")

# Access linked programs directly as Program objects
expert_program = main_program.linked_programs["expert"]
print(f"Expert model: {expert_program.model_name}")

# Linked programs also have their own linked programs
if "utility" in expert_program.linked_programs:
    utility_program = expert_program.linked_programs["utility"]
    print(f"Utility model: {utility_program.model_name}")
```

## Lazy Instantiation

Compiled programs are instantiated as processes only when needed:

```python
from llmproc import LLMProcess

# Create the main process
main_process = LLMProcess.from_toml("path/to/main.toml")

# The linked programs are stored as Program objects
# They will be instantiated as LLMProcess objects only when used via the spawn tool
response = await main_process.run("Use the expert to analyze this code.")
```

## Implementation Details

- Uses Pydantic models for validation
- Resolves relative file paths based on the TOML file location
- Extracts API parameters for convenient access
- Provides both direct compilation and instantiation methods
- Uses a singleton registry to avoid redundant compilation
- Implements non-recursive BFS for program graph traversal
- Implements two-phase compilation:
  1. First phase: Compile all programs in the graph
  2. Second phase: Update references to link Program objects directly
- Handles circular dependencies gracefully
- Provides better error messages with file path information
- Supports skipping file existence checks for validation-only scenarios