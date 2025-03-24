# File Descriptor System for LLMProc

## Background & Problem

Tools can return large outputs, which can lead to contex exhaustion, and affecting the focus of the model.

Previously, we observed:
- Tool use can return long results, especially with MCP tools
- Certain files and other content can be too large for direct inclusion in context
- Agent frameworks like Cursor and Claude Code implement custom paged file reading

## Solution: File Descriptor System

A kernel-level file descriptor system for managing large tool outputs:

1. Large content (from tools or user) is stored in a file descriptor (fd:1234)
2. A summary with the fd reference and preview is returned instead of full content
3. Content can be read in pages or accessed in full via read_fd tool

## Core Components

1. **File Descriptor Manager** - Maintains references to stored content
2. **Read System Call** - Standard interface for accessing paginated content
3. **Automatic Resource Management** - Handles expiration and cleanup of unused descriptors

## Key Features

1. **Line-Aware Pagination**: Breaks content at line boundaries when possible
2. **Continuation Indicators**: Clear markers when content continues across pages
3. **Character-Based Fallback**: Falls back to character-based pagination for long lines
4. **XML Formatting**: Structured format with metadata for pagination status

## API Design

### 1. File Descriptor Creation (Automatic)

When a tool returns large output:

```python
# Result returned to LLM - XML-formatted response
"""
<fd_result fd="fd:12345" pages="5" truncated="true" lines="1-42" total_lines="210">
  <message>Output exceeds 4000 characters. Use read_fd to read more pages.</message>
  <preview>
  First page content is included here for immediate use. If the preview ends with
  a partial line that continues on the next page, the "truncated" attribute will be true.
  </preview>
</fd_result>
"""
```

### 2. Read File Descriptor

```python
# Input with page number
read_fd(fd="fd:12345", page=2)

# Read the entire file content
read_fd(fd="fd:12345", read_all=True)

# Alternative (for specific line ranges - optional)
read_fd(fd="fd:12345", start_line=45, end_line=90)

# Output - XML-formatted response
"""
<fd_content fd="fd:12345" page="2" pages="5" continued="true" truncated="true" lines="43-84" total_lines="210">
Page content goes here. If this page starts with a continued line from the previous page, 
the "continued" attribute will be true. If this page ends with a truncated line that
continues on the next page, the "truncated" attribute will be true.
</fd_content>
"""
```

The XML format clearly separates content from metadata and makes it easy for the LLM to understand the structure of the response. The attributes provide all necessary information without cluttering the response.

### Note on File Descriptor Lifecycle

We explicitly decided NOT to implement a close_fd system call for the following reasons:

1. **Simplicity**: File descriptors are treated as persistent state, just like conversation history.
2. **Consistency**: FDs should remain available throughout the process lifetime.
3. **Mental Model**: Makes the FD system simpler for LLMs to understand and use.
4. **Cross-Process Behavior**: Simplifies inheritance during fork operations.
5. **Error Prevention**: Avoids issues with dangling references to closed FDs.

File descriptors will naturally be cleaned up when a process ends, and any memory constraints would be addressed through the future disk offloading system rather than manual resource management.

### 3. FD to File Operation

```python
# Write FD content to a file
fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", mode="write")

# Write specific page
fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", page=2, mode="write")

# Write specific line range
fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", start_line=45, end_line=90, mode="write")

# Append content to existing file
fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", mode="append")

# Insert at specific line
fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", insert_at_line=100, mode="insert")

# Output - XML-formatted response
"""
<fd_write fd="fd:12345" file_path="/path/to/output.txt" success="true" mode="write">
  <message>Content from fd:12345 successfully written to /path/to/output.txt</message>
  <stats>
    <bytes>25600</bytes>
    <lines>320</lines>
  </stats>
</fd_write>
"""
```

## Implementation Details

### File Descriptor Structure

```python
{
  "content": str,          # Full content
  "lines": list[int],      # Start indices of each line
  "total_lines": int,      # Total line count
  "page_size": int,        # Characters per page
  "creation_time": str,    # Creation timestamp
  "source": str            # Origin of content (e.g., "tool:github_search", "user_input")
}
```

### Integration with LLMProcess

1. Add FileDescriptorManager to LLMProcess state
2. Update fork_process to copy file descriptors
3. Add automatic wrapping of large tool outputs
4. Add automatic wrapping of large user inputs (stdin)
5. Register read_fd and fd_to_file tools
6. Add system prompt instructions about file descriptor usage

### Persistence Model

File descriptors are treated as persistent state:

1. **Full Persistence**:
   - FDs remain available indefinitely, just like conversation state
   - No automatic expiration or cleanup
   - FDs are only lost if explicitly cleared or when the process ends

2. **Future Enhancement: Disk Offloading**:
   - Combined checkpoint system for both state and FDs
   - Automatically offload to disk when memory pressure is high
   - Create recovery points to restore full process state with FDs
   - Enable process hibernation and restoration

## FD Management Strategy

The file descriptor system treats file descriptors as first-class conversation state:

1. **Persistence First**:
   - FDs are persistent by default and remain available throughout process lifetime
   - FDs form part of the process's "memory" just like conversation history
   - This approach simplifies usage and avoids lost references

2. **Process Lifecycle Integration**:
   - FDs are fully copied during fork operations
   - Shared FDs can be explicitly preloaded into child processes through spawn
   - FDs naturally end with process termination

3. **Future: Checkpointing and Recovery**:
   - Combined checkpointing of conversation state and file descriptors
   - Enable hibernation/resumption of processes with all their FDs
   - Recovery from crashes with full context restoration

## Configuration

Simple TOML configuration:

```toml
[file_descriptor]
enabled = true                      # Enable file descriptor system
max_direct_output_chars = 8000      # Threshold for FD creation (larger than page size)
default_page_size = 4000            # Default page size for pagination
max_input_chars = 8000              # Threshold for creating FD from user input
page_user_input = true              # Whether to enable paging for user inputs
```

### Sizing Strategy

The relationship between configuration values is important:

1. `max_direct_output_chars` should generally be larger than `default_page_size` 
   - This ensures the first page fits within context without pagination
   - Recommended ratio: max_direct_output_chars = 1.5-2x default_page_size

2. `max_input_chars` should be set based on typical user input patterns
   - Lower values improve memory usage but increase pagination frequency
   - Higher values preserve more context in a single turn but use more memory

## Example Usage

```
Human> Search for quantum computing papers

Assistant> I'll search for quantum computing papers.

[Uses web_search tool and gets a large result]

I found information about quantum computing papers. The results are extensive,
so they've been stored in a file descriptor.

<fd_result fd="fd:12345" pages="5" truncated="false" lines="1-42" total_lines="210">
  <message>Output exceeds 4000 characters. Use read_fd to read more pages.</message>
  <preview>
  Recent papers in quantum computing have shown remarkable progress in several areas...
  [preview content]
  </preview>
</fd_result>

I'll read more information to provide a complete answer.

read_fd(fd="fd:12345", page=2)

<fd_content fd="fd:12345" page="2" pages="5" continued="false" truncated="false" lines="43-84" total_lines="210">
Quantum algorithms have also seen significant improvements...
[page 2 content]
</fd_content>

Based on all the information I've found, here's a summary of key quantum computing papers...
```

For more detailed examples and advanced usage patterns, see `fd_implementation_phases.md`.

## User Input Handling

The file descriptor system automatically manages large user inputs:

1. When a user inputs content exceeding the `max_input_chars` threshold:
   - The content is stored in a file descriptor
   - The message is replaced with an FD reference and preview

2. Example:
   ```
   Human> [Sends a 15,000 character log file]

   # Automatically transformed to:
   <fd_result fd="fd:9876" pages="4" truncated="false" lines="1-320" total_lines="320">
     <message>Large user input has been stored in a file descriptor.</message>
     <preview>[First few lines of content...]</preview>
   </fd_result>

   Assistant> I see you've shared a log file. Let me read it.
   read_fd(fd="fd:9876", page=1)
   ```

For more detailed implementation plans, see `fd_implementation_phases.md`.

## Cross-Process Behavior

File descriptors work with multi-process features:

1. **Automatic Inheritance with Fork**:
   - FDs are automatically inherited during `fork`
   - Child processes access parent's FDs with the same IDs

2. **Explicit Sharing with Spawn**:
   - Parent can use `additional_preload_fds` parameter to share specific FDs
   - Content appears in child's enriched system prompt

## Benefits

- No truncation of large tool outputs or user inputs
- Consistent interface for accessing large content
- Tools don't need custom pagination logic
- XML format clearly communicates pagination status
- LLM can choose how much content to read
- Efficient cross-process content sharing

## System Prompt Additions

Instructions added to system prompt when enabled:

```
<file_descriptor_instructions>
This system includes a file descriptor feature for handling large content:

1. Large outputs are stored in file descriptors (fd:12345)
2. Use read_fd to access content in pages or all at once
3. Use fd_to_file to export content to files

Key commands:
- read_fd(fd="fd:12345", page=2) - Read page 2
- read_fd(fd="fd:12345", read_all=True) - Read entire content
- fd_to_file(fd="fd:12345", file_path="output.txt") - Write to file

Tips:
- Check "truncated" and "continued" attributes for content continuation
- When analyzing large content, consider reading all pages first
- For very large content, consider using fork to delegate analysis
</file_descriptor_instructions>
```

## Feature Configuration

The file descriptor system is configured through two complementary mechanisms:

1. **Basic enablement** through the `[tools]` section:

```toml
[tools]
# Enable file descriptor-related tools
enabled = ["read_fd"]                       # Basic FD reading capability
enabled = ["read_fd", "fd_to_file"]         # Enable file export capability
enabled = ["read_fd", "fd_to_file", "fork"] # Enable FD features with forking
```

2. **Advanced settings** in the `[file_descriptor]` section:

```toml
[file_descriptor]
enabled = true                     # Explicit enable flag (optional)
max_direct_output_chars = 8000     # Threshold for FD creation (larger than page size)
default_page_size = 4000           # Page size for pagination
max_input_chars = 8000             # Threshold for user input FD creation
page_user_input = true             # Enable/disable user input paging
```

Configuration notes:
- The file descriptor system is enabled if:
  - Any FD tool is included in `[tools].enabled` (like "read_fd") OR
  - `enabled = true` exists in the `[file_descriptor]` section
- For full functionality, you need both the system enabled AND appropriate tools in `[tools].enabled`
- At compile time:
  - A warning is issued if `[file_descriptor]` has configuration but no FD tools are enabled
  - An error is raised if `[file_descriptor].enabled = true` but no FD tools are in `[tools].enabled`
- Additional features like "fd_to_file" follow the same pattern as other tools
- Security-sensitive features can be disabled by omitting them from enabled tools

## Error Handling

The FD system handles errors gracefully to avoid disrupting the conversation:

1. **Invalid FD References**:
   ```python
   # When referencing a non-existent FD
   read_fd(fd="fd:nonexistent")
   
   # Returns error message in standardized XML format:
   <fd_error type="not_found" fd="fd:nonexistent">
     <message>File descriptor fd:nonexistent not found</message>
   </fd_error>
   ```

2. **Pagination Errors**:
   ```python
   # When requesting a page beyond available content
   read_fd(fd="fd:12345", page=100)  # But FD only has 5 pages
   
   # Returns error with available range:
   <fd_error type="invalid_page" fd="fd:12345">
     <message>Invalid page number. Valid range: 1-5</message>
   </fd_error>
   ```

3. **File Operation Errors**:
   ```python
   # When file operations fail (e.g., permission denied)
   fd_to_file(fd="fd:12345", file_path="/root/protected.txt")
   
   # Returns detailed error:
   <fd_error type="file_operation" fd="fd:12345">
     <message>Cannot write to file: Permission denied</message>
     <details>Operation: write, Path: /root/protected.txt</details>
   </fd_error>
   ```

All errors follow a consistent XML format to make them easily identifiable and processable by the LLM.

## Testing Strategy

Testing for the FD system should focus on these key areas:

1. **Threshold Boundaries**:
   - Test content exactly at threshold values (max_direct_output_chars, max_input_chars)
   - Test content just above and below thresholds

2. **Content Variety**:
   - Multi-line text (e.g., code, logs)
   - Single-line content (e.g., long JSON)
   - Binary data (should be handled correctly as base64)
   - Unicode and special characters

3. **Pagination Edge Cases**:
   - Content with very long individual lines
   - Content with line breaks at page boundaries
   - Empty content or single-line content

4. **Cross-Process Testing**:
   - FD inheritance during fork
   - FD preloading during spawn
   - References to the same FD from different processes

5. **Integration Testing**:
   - End-to-end tests with real LLM interactions
   - Conversation scenarios with multiple FD references

## Extension Strategy

Guidelines for future extensions:

1. **Feature Detection**:
   - LLMs should detect available features based on system prompt instructions
   - New capabilities should be clearly documented in system prompt
   - Tools should be explicitly enabled in the [tools] section

2. **Compatibility**:
   - New parameters should be optional with reasonable defaults
   - New operations should have distinct names rather than overloading existing ones
   - XML formats should maintain backward compatibility

## Implementation Plan

See `fd_implementation_phases.md` for a detailed phased implementation plan that breaks down the work into clear milestones.

1. Basic file descriptor management (create, read)
2. Line-aware pagination 
3. Large user input detection and wrapping
4. Automatic wrapping for large tool outputs
5. System prompt enrichment with FD instructions
6. Integration with fork system call

## Future Enhancements (TODO)

1. **Disk Offloading and Checkpointing**: Implement combined checkpointing system for both state and FDs
   ```python
   # Create a checkpoint of current process state with all FDs
   checkpoint_id = checkpoint_process()
   
   # Restore a process from checkpoint
   restore_process(checkpoint_id)
   
   # List available checkpoints
   list_checkpoints()
   ```
   - Enables persistent storage of large content beyond process lifetime
   - Provides crash recovery with complete context restoration
   - Supports hibernation of inactive processes to free memory
   - Unified approach that treats conversation state and FDs as a single unit

2. **Universal Conversation FD References**: Automatically assign a unique file descriptor to every message and tool result
   ```python
   # Each message and tool result would get a unique FD
   # Example conversation state with automatic FD assignment:
   [
     {"role": "user", "content": "...", "fd": "fd-1001"},
     {"role": "assistant", "content": "...", "fd": "fd-1002"},
     {"role": "user", "content": [{"type": "tool_result", ...}], "fd": "fd-1003"},
     # ...
   ]
   
   # These FDs could be referenced in spawn or fork
   spawn(
     program="code_expert", 
     query="Fix the bug described in fd-1001 using the error stack trace in fd-1003",
     additional_preload_fds=["fd-1001", "fd-1003"]
   )
   ```
   
   **Benefits**:
   - Creates a complete referenceable "memory" of the conversation
   - Avoids duplicating content in cross-process communication
   - Enables precise referencing of specific parts of the conversation history
   - Makes all conversation history addressable through a consistent interface

2. **Search Capability**: Add `search_fd(fd, query)` function to find specific text patterns without reading the entire content
   ```python
   search_fd(fd="fd-12345", query="ERROR", case_sensitive=False)
   # Returns matching lines with context and position information
   ```

3. **User Message Paging Configuration**:
   ```toml
   [file_descriptor]
   # ...
   page_user_input = true  # Allow disabling user input paging
   ```

4. **Semantic Navigation**: Support navigation by semantic units
   ```python
   read_fd(fd="fd-12345", unit="paragraph", index=3)  # Read third paragraph
   read_fd(fd="fd-12345", unit="function", name="process_data")  # For code files
   ```

5. **Named References**: Allow descriptive naming of file descriptors
   ```python
   rename_fd(fd="fd-12345", name="error_logs")
   read_fd(name="error_logs", page=2)  # Use name instead of fd ID
   ```

<!-- Batch operations removed as they rely on close_fd which has been deprecated -->

7. **Section Referencing System**: Allow marking and retrieving specific sections of content
   ```python
   # Mark a section with a reference ID for later access
   mark_fd_section(fd="fd-12345", start_line=45, end_line=52, ref_id="bug_details")
   
   # Retrieve a specific section
   get_fd_section(fd="fd-12345", ref_id="bug_details")
   
   # List all marked sections in a file descriptor
   list_fd_sections(fd="fd-12345")
   
   # Delete a section reference
   delete_fd_section(fd="fd-12345", ref_id="bug_details")
   ```
   
   **Enhanced Process Communication with File Descriptors**:
   ```python
   # Extended spawn tool with support for different preloading options
   spawn(
     program="sql_expert", 
     query="Optimize the SQL query in the logs (see preloaded content)",
     additional_preload_files=["data/queries.sql"],  # Regular files from filesystem
     additional_preload_fds=["fd-12345"]            # Content from FDs preloaded as full text
   )
   
   # For fork, FDs are automatically inherited by child processes
   fork([
     "Review the error logs in fd-12345, focusing on lines 100-150",
     "Check the config file in fd-67890, looking for security settings"
   ])
   ```
   
   **Advanced Cross-Process FD Features**:
   - Automatic FD inheritance for child processes
   - Options for selective FD sharing with child processes
   - Support for FD metadata that explains content type and structure to child processes

8. **Auto-Summarization**: Generate automatic summaries of file descriptor content
   - Would require an additional LLM call to create the summary
   - Could provide immediate high-level understanding of large content
   
9. **Temporary Shared File System**: Create temporary files accessible to multiple processes
   ```python
   # Create a temporary file that persists for the kernel session
   temp_file_id = create_temp_file("This is shared content for multiple children")
   
   # Use in spawn to provide to multiple children
   spawn(program="child1", query="Analyze this data", additional_preload_temp_files=[temp_file_id])
   spawn(program="child2", query="Summarize this data", additional_preload_temp_files=[temp_file_id])
   
   # Read from a temp file
   read_temp_file(temp_file_id)
   
   # Delete when no longer needed
   delete_temp_file(temp_file_id)
   ```
   - Provides a mechanism for sharing writable content between multiple processes
   - Separate and complementary to the read-only file descriptor system
   - More appropriate for collaborative content creation between processes
