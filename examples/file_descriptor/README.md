# File Descriptor Examples

This directory contains examples demonstrating the file descriptor system in llmproc.

## Overview

The file descriptor system enables:
1. Handling content that exceeds context limits
2. Sharing large content between linked processes
3. Advanced content positioning and extraction
4. Managing large user inputs

## Examples

- `main.toml`: Core file descriptor features (read_fd, fd_to_file, read_file)
- `spawn_integration.toml`: Sharing file descriptors between processes
- `analyzer.toml`: Child process for content analysis (used with spawn_integration)
- `transformer.toml`: Child process for content transformation (used with spawn_integration)
- `user_input.toml`: Handling large user inputs with automatic FD creation
- `references.toml`: Response reference ID system for marking and exporting content

## Running Examples

```bash
# Basic file descriptor features
llmproc-demo ./examples/file_descriptor/main.toml

# File descriptor with spawn integration
llmproc-demo ./examples/file_descriptor/spawn_integration.toml

# User input handling
llmproc-demo ./examples/file_descriptor/user_input.toml

# Response reference ID system
llmproc-demo ./examples/file_descriptor/references.toml
```

## Key Features Demonstrated

1. **Basic Operations**
   - Reading by page: `read_fd(fd="fd:1", start=0)`
   - Reading all content: `read_fd(fd="fd:1", read_all=true)`
   - Exporting to file: `fd_to_file(fd="fd:1", file_path="output.txt")`

2. **Advanced Positioning**
   - Page-based: `read_fd(fd="fd:1", mode="page", start=2, count=1)`
   - Line-based: `read_fd(fd="fd:1", mode="line", start=10, count=5)`
   - Character-based: `read_fd(fd="fd:1", mode="char", start=100, count=50)`

3. **Content Extraction**
   - Creating new FDs from portions: `read_fd(fd="fd:1", extract_to_new_fd=true)`
   - Extracting specific ranges: `read_fd(fd="fd:1", mode="line", start=10, count=5, extract_to_new_fd=true)`

4. **Process Integration**
   - Sharing FDs between processes: `spawn(program_name="analyzer", additional_preload_fds=["fd:1"])`
   - Specialized content processing by child processes

5. **User Input Handling**
   - Automatic FD creation for large inputs
   - Preview with metadata for large inputs

6. **Response References**
   - Marking specific content with `<ref id="example">content</ref>` tags
   - Automatically creating file descriptors from references
   - Accessing references using standard FD tools: `read_fd(fd="ref:example")`
   - Exporting to files: `fd_to_file(fd="ref:example", file_path="output.txt")`