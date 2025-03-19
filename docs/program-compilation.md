# Program Compilation and Linking

This document describes the program compilation and linking system in LLMProc. The system is responsible for:
1. Loading, validating, and processing TOML program files
2. Compiling all linked programs recursively
3. Establishing connections between programs for runtime interaction

## Compilation Process

### Single Program Compilation

When compiling a single program file, the system performs the following steps:

1. **Load and Parse TOML**: The program file is loaded and parsed using the `tomllib` module.
2. **Validate Program**: The parsed program is validated using Pydantic models to ensure it follows the expected schema.
3. **Resolve File Paths**:
   - System prompt files are loaded and validated
   - Preload files are resolved (with warnings for missing files)
   - MCP configuration files are verified
   - Tool settings are extracted
4. **Create Program Instance**: A `LLMProgram` instance is created with the validated program definition.

```python
# Compile a single program
program = LLMProgram.compile("path/to/program.toml")
```

### Recursive Program Compilation

When programs reference other programs through the `[linked_programs]` section, the system can compile all referenced programs recursively:

1. **Traverse Program Graph**: Starting from the main program, the system builds a graph of all linked programs.
2. **Compile Each Program**: Each program in the graph is compiled exactly once, even if referenced multiple times.
3. **Handle Circular Dependencies**: The system detects and correctly handles circular dependencies in the program graph.
4. **Map Programs by Path**: Compiled programs are stored in a dictionary mapping absolute file paths to program instances.

```python
# Compile a main program and all its linked programs
compiled_programs = LLMProgram.compile_all("path/to/main.toml")
```

## Linking Process

After compilation, programs need to be linked together to establish runtime connections. The linking process:

1. **Create Process Instances**: Each compiled program is instantiated as an `LLMProcess`.
2. **Establish Connections**: References between programs are resolved and connected.
3. **Initialize Tools**: Spawn tools and other tools are initialized based on the program settings.

The two-step factory pattern handles the complete compilation and linking process:

```python
# Step 1: Compile the main program and all its linked programs
program = LLMProgram.from_toml("path/to/main.toml")

# Step 2: Start the process
process = await program.start()  # Use await in async context
```

## Program Configuration

Programs are defined in TOML files with standard sections:

```toml
[model]
name = "model-name"
provider = "model-provider"

[prompt]
system_prompt = "System instructions for the model"

[linked_programs]
helper = "path/to/helper.toml"
math = "path/to/math.toml"

[tools]
enabled = ["spawn"]
```

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
compiled_programs = LLMProgram.compile_all("path/to/main.toml")
```

Returns a dictionary mapping absolute file paths to compiled program instances.

### LLMProcess.from_toml

Compiles and links a main program and all its linked programs:

```python
process = LLMProcess.from_toml("path/to/main.toml")
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

Check that your TOML file follows the expected schema. Common issues include:
- Missing required sections or fields
- Incorrect types or formats
- Invalid values for fields

### "Linked program file not found"

Ensure that all referenced program files exist and the paths are correct, especially when using relative paths.

### "Maximum linked program depth exceeded"

Your program graph may have too many levels of nesting or an unintended circular dependency. Try to flatten your program structure.

## Example

Here's a complete example of a program graph:

**main.toml**:
```toml
[model]
name = "main-model"
provider = "anthropic"

[prompt]
system_prompt = "Main program"

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
from llmproc import LLMProcess
process = LLMProcess.from_toml("main.toml")
```

Now you can use the process to run queries, and it will automatically handle spawning to linked programs as needed.