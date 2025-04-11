"""Constants and definitions for the file descriptor system.

This module contains tool definitions, descriptions, and system prompt instructions
for the file descriptor system.
"""

# Tool descriptions - simplified and focused
READ_FD_DESCRIPTION = """
Reads content from a file descriptor (FD) that stores large tool outputs.

Usage:
  read_fd(fd="fd:12345", start=2) - Read page 2 from the file descriptor
  read_fd(fd="fd:12345", read_all=True) - Read the entire content
  read_fd(fd="fd:12345", mode="line", start=10, count=5) - Read lines 10-14
  read_fd(fd="fd:12345", extract_to_new_fd=True) - Create a new FD from content

Parameters:
  fd: File descriptor ID to read from (e.g., "fd:12345" or "ref:example_id")
  read_all: If true, returns the entire content
  extract_to_new_fd: If true, extracts to a new file descriptor
  mode: Positioning mode: "page" (default), "line", or "char"
  start: Starting position (page, line, or character)
  count: Number of units to read (pages, lines, or characters)
"""

FD_TO_FILE_DESCRIPTION = """
Writes file descriptor content to a file on disk.

Usage:
  fd_to_file(fd="fd:12345", file_path="/Users/username/output.txt") - Write to file
  fd_to_file(fd="fd:12345", file_path="/Users/username/output.txt", mode="append") - Append

Parameters:
  fd: File descriptor ID to export (e.g., "fd:12345" or "ref:example_id")
  file_path: Absolute path to the file to write
  mode: "write" (default) or "append"
  create: Create file if it doesn't exist (default: true)
  exist_ok: Allow overwriting existing file (default: true)
"""

# Tool definitions
READ_FD_TOOL_DEF = {
    "name": "read_fd",
    "description": READ_FD_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "fd": {
                "type": "string",
                "description": "The file descriptor ID to read from (e.g., 'fd:12345' or 'ref:example_id')",
            },
            "read_all": {
                "type": "boolean",
                "description": "If true, returns the entire content (use cautiously with very large content)",
            },
            "extract_to_new_fd": {
                "type": "boolean",
                "description": "If true, extracts the content to a new file descriptor and returns the new FD ID",
            },
            "mode": {
                "type": "string",
                "enum": ["page", "line", "char"],
                "description": "Positioning mode: 'page' (default), 'line', or 'char'",
            },
            "start": {
                "type": "integer",
                "description": "Starting position in the specified mode's units (page number, line number, or character position)",
            },
            "count": {
                "type": "integer",
                "description": "Number of units to read (pages, lines, or characters)",
            },
        },
        "required": ["fd"],
    },
}

FD_TO_FILE_TOOL_DEF = {
    "name": "fd_to_file",
    "description": FD_TO_FILE_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "fd": {
                "type": "string",
                "description": "The file descriptor ID to export (e.g., 'fd:12345' or 'ref:example_id')",
            },
            "file_path": {
                "type": "string",
                "description": "The path to the file to write",
            },
            "mode": {
                "type": "string",
                "enum": ["write", "append"],
                "description": "Whether to overwrite ('write', default) or append ('append') to the file",
            },
            "create": {
                "type": "boolean",
                "description": "Whether to create the file if it doesn't exist (default: true)",
            },
            "exist_ok": {
                "type": "boolean",
                "description": "Whether it's ok if the file already exists (default: true)",
            },
        },
        "required": ["fd", "file_path"],
    },
}

# System prompt instructions - streamlined
FILE_DESCRIPTOR_INSTRUCTIONS = """
<file_descriptor_instructions>
This system includes a file descriptor feature for handling large content:

1. Large tool outputs are stored in file descriptors (fd:12345)
2. Use read_fd to access content in pages or all at once
3. Use fd_to_file to export content to disk files

Key commands:
- read_fd(fd="fd:12345", start=2) - Read page 2
- read_fd(fd="fd:12345", read_all=True) - Read entire content
- read_fd(fd="fd:12345", mode="line", start=10, count=5) - Read lines 10-14
- read_fd(fd="fd:12345", mode="char", start=100, count=200) - Read 200 characters
- fd_to_file(fd="fd:12345", file_path="/Users/username/output.txt") - Save to file
- fd_to_file(fd="fd:12345", file_path="/Users/username/output.txt", mode="append") - Append to file

Tips:
- Use mode="line" to read specific lines of content
- Use mode="char" to read specific character ranges
- Use mode="page" with count>1 to read multiple pages at once
- Use extract_to_new_fd=True to create a new file descriptor
</file_descriptor_instructions>
"""

# User input paging
USER_INPUT_INSTRUCTIONS = """
<fd_user_input_instructions>
This system handles large user inputs through file descriptors:

- Large messages are automatically converted to file descriptors
- You'll see a preview: <fd:12345 preview="..." type="user_input" size="10000">
- To read the content: read_fd(fd="fd:1", read_all=true)
- For large inputs:
  * Read sections: read_fd(fd="fd:1", mode="line", start=10, count=5)
  * Extract parts: read_fd(fd="fd:1", extract_to_new_fd=true)
</fd_user_input_instructions>
"""

# Reference system
REFERENCE_INSTRUCTIONS = """
<reference_instructions>
You can mark sections of your responses as references:

<ref id="example_id">
Content here (code, text, data, etc.)
</ref>

These references can be:
- Exported to files: fd_to_file(fd="ref:example_id", file_path="/Users/username/output.txt")
- Read directly: read_fd(fd="ref:example_id", read_all=true)
- Passed to child processes via spawn tool

Use descriptive IDs for your references.
</reference_instructions>
"""

# Registry of FD-related tools that should not trigger recursive FD creation
FD_RELATED_TOOLS = {"read_fd", "fd_to_file"}
