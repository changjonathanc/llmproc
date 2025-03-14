# File Preloading in LLMProc

The preload feature allows you to provide files as context to the LLM at initialization, enhancing the model's ability to reference specific information throughout the conversation.

## How Preloading Works

When you specify files in the `[preload]` section of your TOML configuration, LLMProc will:

1. Read all specified files at initialization time
2. Format the content with XML tags for better context organization
3. Add the content to the system prompt as part of the primary context
4. Maintain this context even after conversation resets (optional)

## TOML Configuration

Add a `[preload]` section to your TOML configuration file:

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

### Configuration-based Preloading

```python
from llmproc import LLMProcess

# Load configuration with preloaded files
process = LLMProcess.from_toml("examples/preload.toml")

# The model already has context from preloaded files
response = process.run("What information can you tell me about the project?")
print(response)  # The response will incorporate information from preloaded files

# Reset conversation but keep preloaded files
process.reset_state(keep_preloaded=True)
```

### Runtime Preloading

```python
from llmproc import LLMProcess

# Start with basic configuration
process = LLMProcess.from_toml("examples/minimal.toml")

# Add files to the conversation context at runtime
process.preload_files([
    "documentation.md",
    "config_schema.json"
])

# Now the model has context from these files
response = process.run("Explain the configuration schema")
print(response)
```

## XML Formatting

Preloaded files are formatted with XML tags for better organization and added to the system prompt:

```xml
You are a helpful assistant...

<preload>
<file path="file1.txt">
Content of file1.txt goes here...
</file>
<file path="file2.md">
Content of file2.md goes here...
</file>
</preload>
```

This format helps the LLM understand the structure and origin of the preloaded content while keeping it within the system context rather than as part of the conversation history.

## Handling Missing Files

If a file specified in the preload section doesn't exist:

1. A warning is printed with XML tags: `<warning>/path/to/missing/file.txt does not exist.</warning>`
2. The process continues, preloading any files that do exist

## Reset Behavior

When resetting the conversation state:

- `reset_state(keep_preloaded=True)`: Keeps preloaded file content in the system prompt
- `reset_state(keep_preloaded=False)`: Removes preloaded content, restoring the original system prompt
- `reset_state(keep_system_prompt=False, keep_preloaded=False)`: Completely resets the conversation
- `reset_state(keep_system_prompt=False, keep_preloaded=True)`: Creates a new system prompt with only preloaded content

## Use Cases

- Providing documentation or reference material for technical questions
- Including project context for more relevant responses
- Supplying background information to maintain conversation relevance
- Loading configuration or settings that should persist across the conversation
- Adding code examples or snippets as reference material

## Testing

The preload feature includes both mechanical tests (checking file loading and state manipulation) and 
functional tests that verify the LLM actually uses the preloaded content:

```bash
# Run regular tests (skips actual API calls)
pytest -v

# Run LLM API tests that verify preloaded content is used in responses
# Requires OPENAI_API_KEY to be set in environment
pytest -v -m llm_api
```

The API test works by:
1. Creating a temporary file with a unique secret flag
2. Preloading this file into the conversation
3. Asking the LLM to identify the secret flag
4. Verifying the LLM's response contains the secret flag

This ensures the preload mechanism is actually providing content that the LLM can access and use.