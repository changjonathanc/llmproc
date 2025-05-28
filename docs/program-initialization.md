# Program Initialization and Linking

This document describes the program initialization and linking system in LLMProc. The system is responsible for:
1. Loading, validating, and processing configuration files (TOML or YAML)
2. Initializing all linked programs recursively
3. Establishing connections between programs for runtime interaction

> **Note**: While the compilation API (`compile()`, `compile_all()`) is valid and used internally, for most applications we recommend using the simpler workflow with `program = LLMProgram.from_file()` followed by `process = await program.start()`. The compilation API may be expanded in the future for advanced use cases like program serialization.

## Initialization Process

### Single Program Initialization

When initializing a single program file, the system performs the following steps:

1. **Load and Parse File**: The program file is loaded and parsed using `tomllib` for TOML or `PyYAML` for YAML files.
2. **Validate Program**: The parsed program is validated using Pydantic models to ensure it follows the expected schema.
3. **Resolve File Paths**:
   - System prompt files are loaded and validated
   - Preload files are resolved (with warnings for missing files)
   - MCP configuration files are verified
   - Tool settings are extracted
4. **Create Program Instance**: A `LLMProgram` instance is created with the validated program definition.

```python
# Create a program from a configuration file
program = LLMProgram.from_file("path/to/program.yaml")  # or .toml
```

### Recursive Program Initialization

When programs reference other programs through the `[linked_programs]` section, the system initializes all referenced programs recursively:

1. **Traverse Program Graph**: Starting from the main program, the system builds a graph of all linked programs.
2. **Initialize Each Program**: Each program in the graph is initialized exactly once, even if referenced multiple times.
3. **Handle Circular Dependencies**: The system detects and correctly handles circular dependencies in the program graph.
4. **Map Programs by Path**: Initialized programs are stored in a dictionary mapping absolute file paths to program instances.

```python
# The start() method automatically handles linked programs
program = LLMProgram.from_file("path/to/main.yaml")  # or .toml
process = await program.start()
```

## Linking Process

After compilation, programs need to be linked together to establish runtime connections. The linking process:

1. **Create Process Instances**: Each compiled program is instantiated as an `LLMProcess`.
2. **Establish Connections**: References between programs are resolved and connected.
3. **Initialize Tools**: Spawn tools and other tools are initialized based on the program settings.

The two-step factory pattern handles the complete compilation and linking process:

```python
# Step 1: Compile the main program and all its linked programs
program = LLMProgram.from_file("path/to/main.yaml")  # or .toml

# Step 2: Start the process
process = await program.start()  # Use await in async context
```

## Program Configuration

Programs are defined in TOML or YAML files with standard sections:

```toml
[model]
name = "model-name"
provider = "model-provider"
max_iterations = 10  # Maximum iterations for tool calls, defaults to 10 if not specified

[prompt]
system_prompt = "System instructions for the model"
user = "Optional user prompt to execute automatically"  # Auto-executes when program starts

[linked_programs]
helper = "path/to/helper.toml"
math = "path/to/math.toml"

[demo]
prompts = [  # Optional list of prompts to execute sequentially
  "First prompt in demo mode",
  "Second prompt in demo mode"
]
pause_between_prompts = true  # Whether to pause between demo prompts, defaults to true

[tools]
enabled = ["spawn"]
```

### User Prompt Section

The `[prompt]` section now includes an optional `user` field to specify an initial user prompt that will be executed automatically when the program starts:

```toml
[prompt]
system_prompt = "System instructions for the model"
user = "What are the key features of LLMProc?"  # Executed automatically when the program starts
```

When a user prompt is specified in the configuration file, it follows a priority order for execution:
1. Command-line prompt argument (via `-p "prompt"` or `--prompt "prompt"`)
2. Standard input (via `cat file.txt | llmproc-demo config.yaml`)
3. File-defined user prompt (via `user = "prompt"` in the config file)
4. Interactive mode (if none of the above is provided)

### Max Iterations Configuration

The `[model]` section includes an optional `max_iterations` field to control the maximum number of tool calls:

```toml
[model]
name = "claude-3-7-sonnet"
provider = "anthropic"
max_iterations = 15  # Allow up to 15 iterations of tool calls
```

If not specified, the default is 10 iterations. This value can be overridden at runtime by passing the `max_iterations` parameter to the `run()` method.

### Demo Mode

The optional `[demo]` section enables running multiple prompts sequentially:

```toml
[demo]
prompts = [
  "What are the key features of LLMProc?",
  "How does program linking work?",
  "Explain the file descriptor system"
]
pause_between_prompts = true  # Pause between prompts for user review
```

When demo mode is enabled, the prompts are executed in sequence, with optional pauses between each prompt. This is useful for demonstrations, tutorials, and testing.

### Linked Programs Section

The `[linked_programs]` section defines connections to other program files:

```toml
[linked_programs]
# Format: name = "path/to/program.toml"
helper = "./helper.toml"
math = "./math.toml"
```

Each entry maps a logical name to a file path. The path can be:
- Relative to the current program file
- Absolute (rarely used)

## Error Handling

The compilation and linking system handles several types of errors:

1. **Missing Files**:
   - Required files (system prompt files, MCP config files, linked program files) raise exceptions
   - Optional files (preload files) generate warnings

2. **Validation Errors**:
   - TOML parsing errors
   - Schema validation errors
   - Type checking errors

3. **Linking Errors**:
   - Missing linked programs
   - Circular dependencies (handled correctly)
   - Maximum recursion depth exceeded

## Debugging

To debug compilation and linking issues:

1. Check warnings during compilation for missing files or other problems.
2. Ensure all referenced files exist and have the correct paths.
3. Verify that the program definition follows the expected schema.
4. Use the `compile_all` method to compile programs separately from linking to isolate issues.

## Implementation Details

### LLMProgram.compile

Compiles a single program file:

```python
program = LLMProgram.compile("path/to/program.toml")
```

### LLMProgram.compile_all

Compiles a main program and all its linked programs recursively:

```python
compiled_programs = LLMProgram.compile_all("path/to/main.yaml")  # or .toml
```

Returns a dictionary mapping absolute file paths to compiled program instances.

### LLMProcess.from_toml

Compiles and links a main program and all its linked programs:

```python
process = LLMProcess.from_file("path/to/main.yaml")  # or .toml
```

Returns an `LLMProcess` instance with all linked programs properly connected.

### LLMProcess._initialize_linked_programs

Initializes linked programs when an `LLMProcess` is created manually:

```python
process = LLMProcess(program=main_program)
process._initialize_linked_programs(linked_programs_dict)
```

## Best Practices

1. **Keep Program Files Simple**: Each program should have a clear, focused purpose.
2. **Use Relative Paths**: Reference linked programs using paths relative to the current program file.
3. **Avoid Deep Nesting**: Keep the program hierarchy relatively flat for better maintainability.
4. **Handle Missing Files**: Be prepared to handle missing linked program files with appropriate fallbacks.
5. **Test Your Program Graph**: Verify that your program graph compiles and links correctly before deployment.

## Common Errors and Solutions

### "Program file not found"

Ensure the specified program file exists and the path is correct.

### "Invalid program"

Check that your configuration file follows the expected schema. Common issues include:
- Missing required sections or fields
- Incorrect types or formats
- Invalid values for fields

### "Linked program file not found"

Ensure that all referenced program files exist and the paths are correct, especially when using relative paths.

### "Maximum linked program depth exceeded"

Your program graph may have too many levels of nesting or an unintended circular dependency. Try to flatten your program structure.

## Examples

### Program Graph Example

Here's a complete example of a program graph:

**main.toml**:
```toml
[model]
name = "main-model"
provider = "anthropic"
max_iterations = 15

[prompt]
system_prompt = "Main program"
user = "Tell me about the LLMProc architecture"

[tools]
enabled = ["spawn"]

[linked_programs]
helper = "helper.toml"
math = "math.toml"
```

**helper.toml**:
```toml
[model]
name = "helper-model"
provider = "anthropic"

[prompt]
system_prompt = "Helper program"

[linked_programs]
utility = "utility.toml"
```

Compile and link the program graph:

```python
# Using the recommended pattern with program.start()
from llmproc import LLMProgram
program = LLMProgram.from_file("main.yaml")  # or .toml
process = await program.start()  # Creates process with all linked programs

# The process will automatically execute the user prompt specified in the configuration file
# No need to call process.run() unless you want to run additional prompts
```

### Demo Mode Example

**demo.toml**:
```toml
[model]
name = "claude-3-7-sonnet"
provider = "anthropic"

[prompt]
system_prompt = "You are an expert on LLMProc architecture."

[demo]
prompts = [
  "What is LLMProc?",
  "How does the program linking feature work?",
  "Explain the file descriptor system",
  "What tools are available in LLMProc?"
]
pause_between_prompts = true

[tools]
enabled = ["calculator", "read_file"]
```

Running the demo:

```bash
# Run the demo with prompts executing sequentially
llmproc-demo demo.toml
```

### Programmatic API Example

You can also set these features programmatically:

```python
from llmproc import LLMProgram

# Create a program with user prompt and max_iterations
program = LLMProgram(
    model_name="claude-3-7-sonnet",
    provider="anthropic",
    user_prompt="What is the LLMProc architecture?",
    max_iterations=15
)

# Or use setter methods
program.set_user_prompt("Explain program linking")
program.set_max_iterations(20)

# Start the process
process = await program.start()

# The program's user prompt will be executed automatically
# If you want to run additional prompts:
result = await process.run("How does the file descriptor system work?")
```

Using these features, you can build sophisticated LLM applications with automatic execution and complex interaction patterns.
