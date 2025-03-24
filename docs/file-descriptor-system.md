# File Descriptor System

The file descriptor system provides a Unix-like mechanism for handling large content (tool outputs, file contents, etc.) that would otherwise exceed the context window limit.

## Overview

When a tool produces output that's too large to fit directly into the context window, the file descriptor system:

1. Stores the content in memory with a simple identifier (fd:1, fd:2, etc.)
2. Returns a preview with the file descriptor reference
3. Allows the LLM to read the full content in pages using the `read_fd` tool

This system is inspired by Unix file descriptors and is designed to be intuitive for LLMs to understand and use.

## Design

For detailed design documentation, see the RFC files:
- [llmproc_file_descriptor.md](/RFC/llmproc_file_descriptor.md): Primary design document
- [fd_implementation_phases.md](/RFC/fd_implementation_phases.md): Implementation phases
- [fd_spawn_integration.md](/RFC/fd_spawn_integration.md): Integration with spawn system call

## Configuration

The file descriptor system is configured through two mechanisms:

### 1. Tool Configuration

```toml
[tools]
enabled = ["read_fd"]  # Basic FD reading capability
```

### 2. File Descriptor Settings

```toml
[file_descriptor]
enabled = true                      # Explicitly enable (also enabled by read_fd in tools)
max_direct_output_chars = 2000      # Threshold for FD creation
default_page_size = 1000            # Size of each page
max_input_chars = 2000              # Threshold for user input FD creation (future)
page_user_input = true              # Enable/disable user input paging (future)
```

**Note**: The system is enabled if any FD tool is in `[tools].enabled` OR `enabled = true` exists in the `[file_descriptor]` section.

## Usage

### Basic File Descriptor Operations

```python
# Read a specific page
read_fd(fd="fd:1", page=2)

# Read the entire content
read_fd(fd="fd:1", read_all=True)
```

### XML Response Format

File descriptor operations use XML formatting for clarity:

```xml
<!-- Initial FD creation result -->
<fd_result fd="fd:1" pages="5" truncated="true" lines="1-42" total_lines="210">
  <message>Output exceeds 2000 characters. Use read_fd to read more pages.</message>
  <preview>
  First page content is included here...
  </preview>
</fd_result>

<!-- Read FD result -->
<fd_content fd="fd:1" page="2" pages="5" continued="true" truncated="true" lines="43-84" total_lines="210">
Second page content goes here...
</fd_content>
```

### Key Features

- **Line-Aware Pagination**: Breaks content at line boundaries when possible
- **Continuation Indicators**: Shows if content continues across pages 
- **Sequential IDs**: Simple fd:1, fd:2, etc. pattern
- **Recursive Protection**: File descriptor tools don't trigger recursive FD creation

## Implementation

The file descriptor system is implemented in `src/llmproc/tools/file_descriptor.py` with these key components:

1. **FileDescriptorManager**: Core class managing creation and access
2. **read_fd Tool**: System call interface for accessing content
3. **XML Formatting**: Standard response format with metadata

### Integration

The file descriptor system integrates with:

- **LLMProcess**: Initializes and maintains FD manager state
- **AnthropicProcessExecutor**: Automatically wraps large outputs
- **fork_process**: Maintains FD state during process forking

## Implementation Status

### Completed (Phase 1)

The file descriptor system has completed Phase 1 implementation, which includes:

- Basic File Descriptor Manager with in-memory storage
- read_fd tool with page-based access
- Line-aware pagination with boundary detection
- Automatic tool output wrapping
- XML formatting with consistent metadata
- System prompt instructions
- Integration with fork system call
- Protection against recursive file descriptor creation

### Future Enhancements (Phases 2-3)

As documented in the RFCs, planned future enhancements include:

1. **fd_to_file Tool**: Export content to filesystem
2. **User Input Wrapping**: Create FDs for large user inputs 
3. **Enhanced Process Integration**: Better interaction with spawn system call