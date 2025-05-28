# File Preloading in LLMProc

The preload feature allows you to provide files as context to the LLM at initialization, enhancing the model's ability to reference specific information throughout the conversation.

## How Preloading Works

When you specify files in the `[preload]` section of your TOML program, LLMProc will:

1. Read all specified files at initialization time
2. Format the content with XML tags for better context organization
3. Add the content to the system prompt as part of the primary context
4. Maintain this context even after conversation resets (optional)

## TOML Program Format

Add a `[preload]` section to your TOML program file:

```toml
[preload]
files = [
  "path/to/file1.txt",
  "path/to/file2.md",
  "path/to/another/file3.json"
]
relative_to = "program"  # or "cwd" to resolve at runtime
```

By default, paths are resolved relative to the program file. Set
`relative_to = "cwd"` to resolve them relative to the current working
directory when the program starts.

## Examples

### Using Preloaded Files with the New API

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Create a program with preloaded files
    program = (
        LLMProgram(
            model_name="claude-3-7-sonnet-20250219",
            provider="anthropic",
            system_prompt="You are a helpful assistant with knowledge about this project.",
            parameters={"max_tokens": 4096}
        )
        .add_preload_file("README.md")
        .add_preload_file("CONTRIBUTING.md")
    )

    # Start the process (handles async initialization)
    process = await program.start()

    # The model already has context from preloaded files
    run_result = await process.run("What information can you tell me about the project?")

    # Get the assistant's response
    response = process.get_last_message()
    print(f"Response: {response}")  # Will incorporate information from preloaded files

    # Run another query
    run_result = await process.run("Tell me more about the project structure")
    response = process.get_last_message()
    print(f"Response: {response}")

# Run the async function
asyncio.run(main())
```

### Runtime Preloading with spawn

Runtime preloading is now only supported through the `spawn` tool, which allows you to pass additional file paths to preload for the child process:

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load a program with linked programs
    program = LLMProgram.from_toml("examples/program-linking/main.toml")
    process = await program.start()

    # Preload files are passed as an argument to the spawn tool
    run_result = await process.run(
        "Use the spawn tool to call the repo_expert program with these additional files: " +
        "README.md, pyproject.toml, and docs/preload-feature.md"
    )

    # The spawn tool will internally pass these files to create_process
    # with the additional_preload_files parameter

    # Get the response
    response = process.get_last_message()
    print(f"Response: {response}")

# Run the async function
asyncio.run(main())
```

The above example will be processed by the LLM, which would use the spawn tool with syntax like:

```
spawn(
  program_name="repo_expert",
  prompt="Please analyze this repository",
  additional_preload_files=["README.md", "pyproject.toml", "docs/preload-feature.md"]
)
```

## Content Format

Preloaded file content is added to the enriched system prompt in this format:

```
<preload>
<file path="README.md">
# Project Title

This is the README file content...
</file>

<file path="example.py">
def example_function():
    return "This is an example"
</file>
</preload>
```

This structure helps the model understand the source of the information and maintain separation between different files.

## Implementation Details

- Files are loaded at initialization time only
- File content is inserted directly into the `enriched_system_prompt`; it is not stored separately
- The enriched system prompt is generated during process creation, combining the original system prompt with preloaded content
- The enriched system prompt is immutable after process creation
- For child processes created with `spawn`, additional files can be preloaded via the `additional_preload_files` parameter
- The preloaded content is immutable after process creation
- Missing files generate warnings but won't cause the initialization to fail
- File paths are resolved relative to the program file unless
  `relative_to = "cwd"` is specified

> **⚠️ API Stability Note:** The `reset_state()` method in LLMProcess is considered experimental and not yet ready for general use. It may be changed or removed in future releases as we refine the process lifecycle management. For now, consider the preloaded content and system prompt to be immutable after process creation.

## Best Practices

1. **Selective Loading**: Only preload files that are essential for the assistant's knowledge
2. **File Size**: Keep individual files relatively small to avoid context overload
3. **Format Selection**: Use text-based formats (Markdown, code, plain text) for best results
4. **Spawn Tool**: Use the spawn tool with `additional_preload_files` to provide context to child processes dynamically
