# System Prompt Examination Tool

The `llmproc-prompt` tool allows you to examine what the enriched system prompt will look like for a given program configuration, without making any API calls. This is useful for debugging and understanding how your program configuration affects the system prompt sent to the LLM.

## Usage

```bash
# Basic usage - print the enriched system prompt for a program
llmproc-prompt ./examples/file_descriptor/references.toml

# Save the output to a file
llmproc-prompt ./examples/file_descriptor/references.toml -o system_prompt.txt

# Skip environment information
llmproc-prompt ./examples/file_descriptor/references.toml --no-env

# Disable color output
llmproc-prompt ./examples/file_descriptor/references.toml --no-color
```

## Features

This tool shows:

1. The complete enriched system prompt that would be sent to the LLM
2. A summary of which sections are included in the prompt
3. Key program configuration details

Each section is color-coded (unless disabled):
- Base system prompt: normal text
- Environment information: green
- File descriptor instructions: magenta
- Reference instructions: blue
- Preloaded file content: yellow

## Command Line Options

```
usage: llmproc-prompt [-h] [--output OUTPUT] [--no-env] [--no-color] program_path

Print the enriched system prompt for a program

positional arguments:
  program_path          Path to the program TOML file

options:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        File to write output to (default: stdout)
  --no-env, -E          Don't include environment information
  --no-color, -C        Don't colorize the output
```

## Example Output

```
===== ENRICHED SYSTEM PROMPT =====

You are a helpful assistant that specializes in creating code examples.

When you create code examples:
- Always wrap code in reference tags for easy export
- Use clear, descriptive reference IDs
- Provide explanation before and after the code
- Mention that the user can export specific references to files

Whenever you generate multiple examples, create a separate reference for each one.

<env>
working_directory: /path/to/current/directory
platform: darwin
date: 2025-04-01
</env>

<file_descriptor_instructions>
This system includes a file descriptor feature for handling large content:

1. Large tool outputs are stored in file descriptors (fd:12345)
2. Large user inputs may also be stored in file descriptors automatically
...
</file_descriptor_instructions>

<reference_instructions>
You can mark sections of your responses using reference tags:

<ref id="example_id">
Your content here (code, text, data, etc.)
</ref>

These references can be:
- Exported to files using: fd_to_file(fd="ref:example_id", file_path="output.txt")
- Read using standard file descriptor tools: read_fd(fd="ref:example_id", read_all=true)

Choose clear, descriptive IDs for your references.
</reference_instructions>

===== SECTIONS SUMMARY =====

- Base System Prompt ✅
- Environment Information ✅
- Preloaded Files ❌
- File Descriptor Instructions ✅
- Reference ID Instructions ✅

===== PROGRAM CONFIGURATION =====

Model: claude-3-5-sonnet-20240620
Provider: anthropic
Display Name: Claude with References

File Descriptor Configuration:
  Enabled: True
  Max Direct Output Chars: 4000
  Default Page Size: 2000
  Max Input Chars: 8000
  Page User Input: True
  Enable References: True

Enabled Tools:
  - read_fd
  - fd_to_file

============================
```

## Use Cases

1. **Debugging Configuration Issues**
   - Check if reference instructions are being included
   - Verify file descriptor settings are correct
   - Ensure environment information is properly formatted

2. **Development and Testing**
   - Examine the effect of configuration changes without making API calls
   - Verify preloaded files are correctly included
   - Ensure system prompts aren't too large

3. **Documentation and Training**
   - Generate examples of what prompts look like for different configurations
   - Help users understand how TOML configuration maps to the system prompt
   - Create documentation for specific configurations
