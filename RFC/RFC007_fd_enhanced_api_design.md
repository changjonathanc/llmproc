# RFC007: Enhanced File Descriptor API Design

This RFC outlines improvements to the File Descriptor API design, focusing on enhanced read and write operations for better usability and Unix-inspired semantics. For the complete system overview, see [RFC001: File Descriptor System for LLMProc](RFC001_file_descriptor_system.md).

## 1. Background

The current file descriptor system implements basic read and write functionality:

```python
# Current read implementation
read_fd(fd="fd:1234", page=2)
read_fd(fd="fd:1234", read_all=True)

# Current write implementation (Phase 2)
fd_to_file(fd="fd:1234", file_path="/path/to/file.txt")
```

While functional, this API can be enhanced to provide more flexibility and power while maintaining simplicity.

## 2. Goals

1. Improve the read_fd API to allow creating new file descriptors from content slices
2. Enhance fd_to_file with clear, self-explanatory file operation modes
3. Maintain backward compatibility with existing code
4. Follow Unix-inspired semantics while using clear, explicit parameters
5. Allow for future extension without breaking changes

## 3. Enhanced read_fd API

```python
read_fd(
    fd="fd:1234",           # Source file descriptor
    read_all=False,         # Read entire content
    extract_to_new_fd=False,# Extract content to a new file descriptor
    
    # Positioning system with unified parameters
    mode="page",            # "page" (default), "line", "char"
    start=1,                # Starting position in the specified mode's units
                            # (page number, line number, or character position)
    count=1,                # Number of units to read (pages, lines, or characters)
)
```

### 3.1 Parameter Naming Rationale

For the positioning parameters, we consulted Claude (an LLM that will be using this API):
- **start/count** was chosen as most intuitive for LLMs to reason with
- Alternative options like **offset/limit** and **position/count** were considered
- Claude confirmed that **start/count** maps most naturally to how LLMs think about content positioning

### 3.2 Key Improvements

1. **New `extract_to_new_fd` parameter**:
   - When `extract_to_new_fd=True`, returns a new file descriptor containing only the requested content
   - Returns only the new FD identifier (not the content itself)
   - Enables "dup"-like functionality to slice and manage content
   - Supports creation of smaller, more manageable content units
   
   Typical workflow:
   1. First read the content using `read_fd` to examine it
   2. Then decide to extract a specific portion to a new FD using `extract_to_new_fd=True`
   3. The new FD ID is returned for future operations

2. **Future extensibility** with mode/start/count parameters:
   - Can be added in future phases without breaking changes
   - Allows more precise content selection (lines, character ranges)
   - Maintain backward compatibility through sensible defaults

### 3.2 Examples

```python
# Current usage (transitioning to unified parameters)
read_fd(fd="fd:1234", start=2)  # Read page 2 (mode="page" by default)

# New functionality
new_fd = read_fd(fd="fd:1234", start=2, extract_to_new_fd=True)
# Returns new fd:1235 containing just page 2

# Advanced positioning (Phase 3)
new_fd = read_fd(fd="fd:1234", mode="line", start=10, count=5, extract_to_new_fd=True)
# Returns new fd with just lines 10-14

# Reading multiple pages
content = read_fd(fd="fd:1234", mode="page", start=2, count=3)
# Returns pages 2, 3, and 4 combined
```

## 4. Enhanced fd_to_file API

```python
fd_to_file(
    fd="fd:1234",                     # Source file descriptor
    file_path="/path/to/file.txt",    # Destination file path
    mode="write",                     # "write" (default) or "append"
    create=True,                      # Create file if doesn't exist (default: True)
    exist_ok=True                    # Allow file to exist (default: True)
)
```

### 4.1 Key Improvements

1. **Clear operation modes**:
   - `mode="write"`: Standard write/overwrite behavior
   - `mode="append"`: Append to existing file without overwriting

2. **Explicit file creation control**:
   - `create=True`: Create file if it doesn't exist (default)
   - `create=False`: Fail if file doesn't exist

3. **Safety parameter**:
   - `exist_ok=False`: Fail if target file already exists
   - `exist_ok=True`: Allow overwriting existing files (default)

### 4.2 Behavior Matrix

| mode    | create | exist_ok | Behavior                                     |
|---------|--------|---------|----------------------------------------------|
| "write" | True   | True    | Create or overwrite (current default)        |
| "write" | True   | False   | Create only if doesn't exist                 |
| "write" | False  | True    | Update existing only                         |
| "append"| True   | True    | Append, create if needed                     |
| "append"| True   | False   | Append only if exists, else create new       |
| "append"| False  | True    | Append to existing only                      |

### 4.3 Examples

```python
# Standard overwrite (current behavior, unchanged)
fd_to_file(fd="fd:1234", file_path="/path/to/file.txt")

# Append to existing file
fd_to_file(fd="fd:1234", file_path="/path/to/file.txt", mode="append")

# Fail if file doesn't exist (update only)
fd_to_file(fd="fd:1234", file_path="/path/to/file.txt", create=False)

# Create new file only (fail if exists)
fd_to_file(fd="fd:1234", file_path="/path/to/file.txt", exist_ok=False)
```

## 5. Error Handling

The enhanced API will use the existing error format with specific error types:

```python
# Error types for fd_to_file
"not_found"       # File descriptor not found
"write_error"     # General file writing error
"file_exists"     # File exists and exist_ok=False
"file_not_found"  # File doesn't exist and create=False
"permission_error"# Permission denied writing file

# Error types for read_fd
"not_found"       # File descriptor not found
"invalid_page"    # Page number out of range
"invalid_range"   # Invalid line/character range
```

## 6. Implementation Plan

### 6.1 Phase 1: Basic Enhancement (Immediate)

1. Update `read_fd` to support `extract_to_new_fd` parameter
   - The parameter was renamed from `create_fd` to more clearly indicate its purpose
   - When `extract_to_new_fd=True`, creates a new file descriptor with the content
   - Returns the new FD identifier (not the content itself)
   - Requires a second read operation to access the content
2. Update `fd_to_file` to support `mode` parameter (write/append)

### 6.2 Phase 2: Full API Implementation

1. Add `create` and `fail_if_exists` parameters to `fd_to_file`
2. Implement full behavior matrix
3. Add more specific error types

### 6.3 Phase 3: Advanced Positioning (Future)

1. Add `mode`, `start`, and `count` parameters to `read_fd`
2. Implement line and character-based positioning

This implementation plan is aligned with the phases outlined in [RFC004: File Descriptor Implementation Phases](RFC004_fd_implementation_phases.md).

## 7. Backward Compatibility

These enhancements maintain backward compatibility:
- All current function calls continue to work without changes
- Default parameters match current behavior
- New functionality is opt-in via new parameters

## 8. System Prompt Updates

The file descriptor system prompt instructions should be updated to include the enhanced functionality:

```
<file_descriptor_instructions>
This system includes a file descriptor feature for handling large content:

1. Large outputs are stored in file descriptors (fd:1234)
2. Use read_fd to access content in pages or all at once
3. Use fd_to_file to export content to disk files

Key commands:
- read_fd(fd="fd:1234", start=2) - Read page 2
- read_fd(fd="fd:1234", read_all=True) - Read entire content
- read_fd(fd="fd:1234", start=2, extract_to_new_fd=True) - Extract page 2 to a new FD
- read_fd(fd="fd:1234", mode="line", start=10, count=5) - Read lines 10-14
- read_fd(fd="fd:1234", mode="char", start=100, count=200) - Read 200 characters starting at position 100
- fd_to_file(fd="fd:1234", file_path="/path/to/output.txt") - Save to file
- fd_to_file(fd="fd:1234", file_path="/path/to/output.txt", mode="append") - Append to file
- fd_to_file(fd="fd:1234", file_path="/path/to/output.txt", exist_ok=False) - Create only if file doesn't exist
- fd_to_file(fd="fd:1234", file_path="/path/to/output.txt", create=False) - Update existing file only

Tips:
- Check "truncated" and "continued" attributes for content continuation
- When analyzing large content, consider reading all pages first
- Use extract_to_new_fd=True when you need to extract specific content
- Use mode="append" when adding to existing files
- Use exist_ok=False to avoid overwriting existing files
- Use create=False when you want to update only existing files
</file_descriptor_instructions>
```

## 9. Usage Examples

### 9.1 Content Slicing

```python
# Parent process has a large log file in fd:12345
# Extract just the error section
error_section_fd = read_fd(fd="fd:12345", start=3, extract_to_new_fd=True)  # This creates fd:12346

# Now send just the error section to a specialized process
spawn(
  program_name="error_analyzer", 
  query="Analyze this error section",
  additional_preload_fds=[error_section_fd]
)
```

### 9.2 File Append Operations

```python
# Append multiple code sections to a file
fd_to_file(fd="ref:imports", file_path="module.py", mode="write")
fd_to_file(fd="ref:class_definition", file_path="module.py", mode="append")
fd_to_file(fd="ref:helper_functions", file_path="module.py", mode="append")
fd_to_file(fd="ref:main_function", file_path="module.py", mode="append")
```

### 9.3 Safe File Operations

```python
# Create a new configuration file, but don't overwrite existing one
try:
    fd_to_file(fd="ref:default_config", file_path="config.json", exist_ok=False)
    print("Created new configuration file")
except:
    print("Configuration file already exists, not overwriting")
```

## 10. References

- [RFC001: File Descriptor System for LLMProc](RFC001_file_descriptor_system.md) - Main specification document
- [RFC003: File Descriptor Implementation Details](RFC003_file_descriptor_implementation.md) - Technical implementation details
- [RFC004: File Descriptor Implementation Phases](RFC004_fd_implementation_phases.md) - Implementation phases and status
- [RFC005: File Descriptor Integration with Spawn Tool](RFC005_fd_spawn_integration.md) - Integration with spawn system
- [RFC006: Response Reference ID System](RFC006_response_reference_id.md) - Integration with reference ID system