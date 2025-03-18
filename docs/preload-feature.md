# File Preloading in LLMProc

The preload feature allows you to provide files as context to the LLM at initialization, enhancing the model's ability to reference specific information throughout the conversation.

## How Preloading Works

When you specify files in the `[preload]` section of your TOML program, LLMProc will:

1. Read all specified files at initialization time
2. Format the content with XML tags for better context organization
3. Add the content to the system prompt as part of the primary context
4. Maintain this context even after conversation resets (optional)

## TOML Program Configuration

Add a `[preload]` section to your TOML program file:

```toml
[preload]
files = [
  "path/to/file1.txt",
  "path/to/file2.md",
  "path/to/another/file3.json"
]
```

File paths are relative to the location of the TOML file.

## Examples

### Using Preloaded Files with the New API

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load and compile program with preloaded files
    program = LLMProgram.from_toml("examples/preload.toml")
    
    # Start the process (handles async initialization)
    process = await program.start()
    
    # The model already has context from preloaded files
    run_result = await process.run("What information can you tell me about the project?")
    
    # Get the assistant's response
    response = process.get_last_message()
    print(f"Response: {response}")  # Will incorporate information from preloaded files
    
    # Reset conversation but keep preloaded files
    process.reset_state(keep_preloaded=True)
    
    # Run another query
    run_result = await process.run("Tell me more about the project structure")
    response = process.get_last_message()
    print(f"Response after reset: {response}")

# Run the async function
asyncio.run(main())
```

### Runtime Preloading

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load a program without preloaded files initially
    program = LLMProgram.from_toml("examples/basic.toml")
    process = await program.start()
    
    # Add preloaded files at runtime
    process.preload_files([
        "README.md",
        "pyproject.toml",
        "docs/preload-feature.md"
    ])
    
    # Now the model has context from the preloaded files
    run_result = await process.run("What files did you just receive?")
    
    # Get the response
    response = process.get_last_message()
    print(f"Response: {response}")
    
    # You can remove preloaded files by resetting state
    process.reset_state(keep_preloaded=False)

# Run the async function
asyncio.run(main())
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

- Files are loaded at initialization time or when added with `preload_files()`
- Content is stored in the `preloaded_content` dictionary
- The enriched system prompt is generated on first run, combining the original system prompt with preloaded content
- File content is preserved across resets unless `keep_preloaded=False` is specified
- Missing files generate warnings but won't cause the initialization to fail
- File paths are resolved relative to the TOML file's location

## Best Practices

1. **Selective Loading**: Only preload files that are essential for the assistant's knowledge
2. **File Size**: Keep individual files relatively small to avoid context overload
3. **Format Selection**: Use text-based formats (Markdown, code, plain text) for best results
4. **Context Reset**: Reset state with `keep_preloaded=True` to maintain file context while clearing conversation history
5. **Dynamic Loading**: Use `preload_files()` to add context based on user queries or session state