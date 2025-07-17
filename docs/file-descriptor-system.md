# File Descriptor System

The file descriptor system provides a Unix-like mechanism for handling large content (tool outputs, file contents, etc.) that would otherwise exceed the context window limit.

## Overview

When a tool produces output that's too large to fit directly into the context window, the file descriptor system:

1. Stores the content in memory with a simple identifier (fd:1, fd:2, etc.)
2. Returns a preview with the file descriptor reference
3. Allows the LLM to read the full content in pages using the `read_fd` tool

This system is inspired by Unix file descriptors and is designed to be intuitive for LLMs to understand and use. For the design rationale behind this approach, see the [API Design FAQ](../FAQ.md#whats-the-purpose-of-the-file-descriptor-system).

## Design

The file descriptor system is designed with several key components:

- **Core File Descriptor System**: The basic system for storing and retrieving large content
- **Implementation Details**: Technical details of the storage and pagination
- **Implementation Phases**: The system was implemented in multiple phases
- **Spawn Tool Integration**: Special integration with the spawn tool for cross-process sharing
- **Response Reference ID System**: System for marking up content with reference IDs
- **Enhanced API Design**: Advanced features for precise content access and manipulation

## Configuration

The file descriptor system is configured through two mechanisms:

### 1. Tool Configuration

```yaml
tools:
  builtin:
    - read_fd
    - fd_to_file  # FD reading and file export capability
```

### 2. File Descriptor Settings

```yaml
plugins:
  file_descriptor:
    enabled: true                      # Explicitly enable the plugin
    max_direct_output_chars: 8000      # Threshold for FD creation
    default_page_size: 4000            # Size of each page
    max_input_chars: 8000              # Threshold for user input FD creation
    page_user_input: true              # Enable/disable user input paging
    enable_references: true            # Enable the reference ID system
```
**Note**: The system is enabled when the plugin is configured. The
`read_fd` and `fd_to_file` tools are provided automatically by the plugin.

## Usage

### Basic File Descriptor Operations

```python
# Basic Operations
read_fd(fd="fd:1", start=2)                # Read page 2
read_fd(fd="fd:1", read_all=True)          # Read the entire content
fd_to_file(fd="fd:1", file_path="/path/to/output.txt")  # Export to file

# Enhanced operations
# Extract content to a new file descriptor
new_fd = read_fd(fd="fd:1", start=2, extract_to_new_fd=True)  # Returns new FD like "fd:2"

# Read specific lines (Enhanced API)
content = read_fd(fd="fd:1", mode="line", start=10, count=5)  # Read lines 10-14

# Read character ranges (Enhanced API)
content = read_fd(fd="fd:1", mode="char", start=100, count=200)  # Read 200 chars

# Append to existing file (Enhanced API)
fd_to_file(fd="fd:1", file_path="/path/to/output.txt", mode="append")

# Control file creation behavior (Enhanced API)
fd_to_file(fd="fd:1", file_path="/path/to/output.txt", exist_ok=False)  # Don't overwrite
fd_to_file(fd="fd:1", file_path="/path/to/output.txt", create=False)    # Update only
```

### Reference ID System

The reference ID system allows the LLM to mark up content in its responses for later reference:

```
Here's a code snippet I want to reference later:

<ref id="example_code">
def hello_world():
    print("Hello, world!")
</ref>

You can use the reference later by accessing it with read_fd(fd="ref:example_code").
```

References are automatically created as file descriptors with the prefix `ref:` followed by the ID.
They can be accessed using the standard `read_fd` tool just like any other file descriptor.

### Spawn and Fork Integration

References are automatically shared between processes during fork and spawn operations:

1. When forking a process, all references from the parent are copied to the child.
2. When spawning a child process, all references from the parent are automatically shared.
3. References created in child processes remain isolated and are not visible to the parent.

This enables complex workflows where multiple processes can share marked-up content without
explicitly passing file descriptors.

```python
# Parent process creates a reference
<ref id="important_data">Data the child will need</ref>

# Child process can access the reference automatically
spawn(program_name="analyzer", prompt="Analyze the content in ref:important_data")

# Inside the child process
read_fd(fd="ref:important_data")  # Works automatically
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

<!-- File export result -->
<fd_file_result fd="fd:1" file_path="/path/to/output.txt" char_count="12345" size_bytes="12345" success="true">
  <message>File descriptor fd:1 content (12345 chars) successfully written to /path/to/output.txt</message>
</fd_file_result>
```

### Key Features

- **Line-Aware Pagination**: Breaks content at line boundaries when possible
- **Continuation Indicators**: Shows if content continues across pages
- **Sequential IDs**: Simple fd:1, fd:2, etc. pattern
- **Recursive Protection**: File descriptor tools don't trigger recursive FD creation
- **Filesystem Export**: Can save file descriptor content to disk files
- **Parent Directory Creation**: Automatically creates directories when exporting to files
- **Reference ID System**: Mark up content with reference IDs for later access
- **Reference Inheritance**: References automatically shared during fork/spawn operations

## Implementation

The file descriptor system is implemented in the `src/llmproc/file_descriptors/` package with these key components:

1. **FileDescriptorManager**: Core class managing creation and access
2. **read_fd Tool**: System call interface for accessing content
3. **fd_to_file Tool**: System call interface for exporting content to files
4. **XML Formatting**: Standard response format with metadata
5. **Reference Extraction**: System for extracting and managing reference IDs

### Integration

The file descriptor system integrates with:

- **LLMProcess**: Initializes and maintains FD manager state
- **AnthropicProcessExecutor**: Automatically wraps large outputs
- **fork_process**: Maintains FD state during process forking
- **spawn_tool**: Shares references between parent and child processes

### Hook System Migration

Starting with version 0.10.0, file descriptor handling is provided through the
plugin-based hook system. The built-in `FileDescriptorPlugin` listens for
`hook_user_input` and `hook_tool_result` events to convert large inputs and
tool outputs into file descriptors. This plugin is registered automatically when
the file descriptor system is enabled, replacing the older embedded logic in
`ToolManager` and `LLMProcess`.

### Performance Optimizations

The file descriptor system includes several performance optimizations:

- **Reference Extraction**: O(n log n) algorithm for detecting nested references
- **Compiled Regex**: Pre-compiled regex patterns for faster reference extraction
- **Proper Pagination**: Uses the calculate_total_pages function for consistent pagination
- **Efficient XML Formatting**: Streamlined XML generation for standard outputs
- **Memory Efficiency**: File descriptors are copied by reference where possible

## Implementation Status

Below is a summary of the implementation status as of 2025-03-25:

### ✅ Phase 1: Core Functionality (Completed)

The file descriptor system has completed Phase 1 implementation, which includes:

- Basic File Descriptor Manager with in-memory storage
- read_fd tool with page-based access
- Line-aware pagination with boundary detection
- Automatic tool output wrapping
- XML formatting with consistent metadata
- System prompt instructions

### ✅ Phase 2: Process Integration (Completed)

- ✅ **Fork Integration**: Automatic FD copying during fork
- ✅ **Cross-Process FD Access**: Implemented via reference inheritance
- ✅ **Spawn Tool Integration**: Implemented with complete support for FD sharing

### ✅ Phase 2.5: API Enhancements (Completed)

- ✅ **Enhanced read_fd**: Implemented with extract_to_new_fd and positioning modes
- ✅ **Enhanced fd_to_file**: Implemented with mode, create, and exist_ok parameters

### 🔄 Phase 3: Optional Features (Partially Completed)

- ✅ **fd_to_file Tool**: Export FD content to filesystem files
- 🔄 **User Input Handling**: Partially implemented (basic user input paging)
- ✅ **Reference ID System**: Complete reference ID system implemented

### Implementation Location

The file descriptor system is implemented in the following files:

- `src/llmproc/file_descriptors/`: Package containing all file descriptor functionality
  - `manager.py`: Core implementation of FileDescriptorManager
  - `plugins/file_descriptor/plugin.py`: FileDescriptorPlugin with read_fd and fd_to_file tool implementations
  - `references.py`: Reference ID system implementation (optimized for performance)
  - `paginator.py`: Line-aware content pagination
  - `formatter.py`: XML formatting utilities
  - `constants.py`: Shared constants and definitions
- `src/llmproc/llm_process.py`: Integration with LLMProcess
- `src/llmproc/providers/anthropic_process_executor.py`: Integration with AnthropicProcessExecutor for output wrapping
 - `src/llmproc/plugins/spawn.py`: Integration with spawn system call for reference inheritance
- `src/llmproc/config/schema.py`: FileDescriptorConfig configuration schema
- Tests:
  - `tests/test_file_descriptor.py`: Basic unit tests
  - `tests/test_file_descriptor_integration.py`: Integration tests with process model
  - `tests/test_fd_to_file_tool.py`: Tests for FD export functionality
  - `tests/test_file_descriptor_core.py`: Core tests including reference ID handling
  - `tests/test_fd_spawn_integration.py`: Tests for spawn integration
  - `tests/test_fd_all_features.py`: Tests for all FD features combined

---
[← Back to Documentation Index](index.md)
