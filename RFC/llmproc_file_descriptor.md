# File Descriptor System for LLMProc

## Background & Problem

Tools often return large outputs that exceed context limits, causing information loss through truncation. Each tool currently needs to implement its own pagination logic, leading to inconsistency and duplicated effort.

Previously, we observed:
- Tool use can return very long results, especially with MCP tools
- Files and other content can be too large for direct inclusion in context
- Agent frameworks like Cursor and Claude Code implement custom paged file reading

## Solution: File Descriptor System

A kernel-level file descriptor system for managing large tool outputs:

1. When output exceeds a threshold, it's stored in a file descriptor
2. The LLM receives a summary with a file descriptor reference
3. The LLM can read content in pages using a standard system call

## Core Components

1. **File Descriptor Manager** - Maintains references to stored content
2. **Read System Call** - Standard interface for accessing paginated content
3. **Automatic Resource Management** - Handles expiration and cleanup of unused descriptors

## Key Features

1. **Line-Aware Pagination**: Breaks content at line boundaries whenever possible
2. **Long Line Handling**: Special handling for content exceeding page size (like large JSON strings)
3. **Continuation Indicators**: Clear markers when lines are continued across pages
4. **Character-Based Fallback**: For single-line content, pagination shifts to character-based
5. **Format-Specific Handling**: Optional JSON pretty-printing for improved readability

## Edge Cases

### Long Line Handling

For content with very long lines (e.g., large JSON strings):

1. **Special Formatting**:
   - Add appropriate XML attributes to indicate line continuation
   - For JSON, optionally pretty-print before pagination for better readability

2. **Example with Long JSON**:
   ```
   # First page (truncated)
   {"results":[{"title":"Quantum Computing","abstract":"Lorem ipsum...

   # Second page (continued and truncated)
   ...dolor sit amet","url":"https://example.com/quantum1"},{"title":"Quantum...

   # With optional pretty-print formatting
   {
     "results": [
       {
         "title": "Quantum Computing",
         "abstract": "Lorem ipsum dolor sit amet",
         "url": "https://example.com/quantum1"
       },
       ...
   ```

## API Design

### 1. File Descriptor Creation (Automatic)

When a tool returns large output:

```python
# Result returned to LLM - XML-formatted response
"""
<fd_result fd="fd-12345" pages="5" truncated="true" lines="1-42" total_lines="210">
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
read_fd(fd="fd-12345", page=2)

# Read the entire file content
read_fd(fd="fd-12345", read_all=True)

# Alternative (for specific line ranges - optional)
read_fd(fd="fd-12345", start_line=45, end_line=90)

# Output - XML-formatted response
"""
<fd_content fd="fd-12345" page="2" pages="5" continued="true" truncated="true" lines="43-84" total_lines="210">
Page content goes here. If this page starts with a continued line from the previous page, 
the "continued" attribute will be true. If this page ends with a truncated line that
continues on the next page, the "truncated" attribute will be true.
</fd_content>
"""
```

The XML format clearly separates content from metadata and makes it easy for the LLM to understand the structure of the response. The attributes provide all necessary information without cluttering the response.

# Note: Explicit close_fd system call removed in favor of automatic management

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
5. Register read_fd tool
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
json_pretty_print = true            # Pretty print JSON for better pagination (optional)
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

### Normal Multi-Line Content

```
Human> Search for quantum computing papers

Assistant> I'll search for quantum computing papers.

[Uses web_search tool and gets a large result]

I found information about quantum computing papers. The results are extensive,
so they've been stored in a file descriptor.

The system returned:
<fd_result fd="fd-12345" pages="5" truncated="false" lines="1-42" total_lines="210">
  <message>Output exceeds 4000 characters. Use read_fd to read more pages.</message>
  <preview>
  Recent papers in quantum computing have shown remarkable progress in several areas. 
  The most significant developments include advances in quantum error correction, 
  quantum machine learning, and quantum algorithm optimization.
  
  A 2023 paper by Johnson et al. demonstrated a new approach to quantum error correction 
  that achieves a 50% reduction in overhead compared to previous methods...
  [additional preview content]
  </preview>
</fd_result>

I'll read more information to provide a complete answer.

[Uses read_fd tool]
read_fd(fd="fd-12345", page=2)

The system returned:
<fd_content fd="fd-12345" page="2" pages="5" continued="false" truncated="false" lines="43-84" total_lines="210">
Quantum algorithms have also seen significant improvements. The work by Chen and 
Smith (2024) presents a novel approach to the quantum traveling salesman problem 
that shows a quadratic speedup over the best known classical algorithms.

Additionally, researchers at MIT have demonstrated a new quantum machine learning 
technique that can process data with significantly fewer qubits than previously 
thought possible...
[additional content]
</fd_content>

Based on all the information I've found, here's a summary of key quantum computing papers...
```

### Long JSON String Example

```
Human> Get detailed data about quantum computing research funding

Assistant> I'll retrieve detailed funding data for quantum computing research.

[Uses funding_data_tool and gets a large JSON result]

I've retrieved the detailed funding data, but it's quite extensive (single JSON object of 28,500 characters),
so it has been stored in a file descriptor.

The system returned:
<fd_result fd="fd-7890" pages="8" truncated="true" lines="partial" total_lines="1">
  <message>Output exceeds 4000 characters. Use read_fd to read more pages.</message>
  <preview>
  {"funding_data":{"total_global_investment":15.2,"currency":"USD_billions","by_country":[{"name":"USA",...
  </preview>
</fd_result>

I'll analyze this data to provide insights about quantum computing research funding.

[Uses read_fd tool]
read_fd(fd="fd-7890", page=2)

The system returned:
<fd_content fd="fd-7890" page="2" pages="8" continued="true" truncated="true" lines="partial" total_lines="1">
...{"name":"China","total_investment":4.8,"major_programs":["Quantum Information Science",...
</fd_content>

After analyzing all the funding data, I can tell you that quantum computing research has seen 
dramatic growth in the past five years, with the USA, China, and EU being the top funders...
```

## User Input Handling

The file descriptor system can also manage large user inputs (stdin):

1. When a user inputs content exceeding the `max_input_chars` threshold:
   - The entire message is automatically stored in a file descriptor
   - The entire user message is replaced with an FD reference and preview

2. Example workflow:
   ```
   Human> [Sends a 15,000 character log file]

   # Automatically transformed to:
   <fd_result fd="fd-9876" pages="4" truncated="false" lines="1-320" total_lines="320">
     <message>Large user input has been stored in a file descriptor. Use read_fd to access the content.</message>
     <preview>
     [2024-03-23 08:15:02] INFO: Application startup
     [2024-03-23 08:15:03] INFO: Loading configuration
     [2024-03-23 08:15:04] INFO: Initializing database connection
     ...
     </preview>
   </fd_result>

   Assistant> I see you've shared a log file. Let me read through it.

   read_fd(fd="fd-9876", page=1)
   ```

3. Benefits:
   - Avoids context bloat from large inputs
   - Preserves conversation history readability
   - Provides a consistent interface for handling both input and output content
   - User doesn't need to explicitly store content in files

4. Future enhancement (TODO):
   - Support chunk-aware paging where the LLMProcess interface could accept a list of message chunks
   - This would allow selective paging of only the large portions of a message
   - Would enable preserving user's introduction text while paging only the large content sections

## Cross-Process Behavior

File descriptors interact with multi-process features in several ways:

1. **Automatic Inheritance**:
   - File descriptors are automatically inherited by child processes during `fork`
   - Child processes can directly access parent's FDs with the same IDs

2. **Explicit Content Sharing**:
   - Parent can use `additional_preload_fds` parameter in `spawn` to:
     - Include content from specific FDs in child's enriched system prompt
     - Make content immediately available without requiring explicit `read_fd` calls

3. **Context Control**:
   - Parent process determines what context is most relevant:
     - Fork provides full access to all parent's FDs
     - Spawn with additional_preload_fds gives precise control over shared content

## Pros and Cons

### Pros
- Users can let LLM processes call any tools without worrying about context limits
- Tools don't need to reimplement pagination features
- XML format makes it easy for LLMs to understand pagination status
- Line-awareness improves readability for multi-line content

### Cons
- Requires more tool calls to read in the context of a long file
- May be more complex than direct truncation for simple cases

## System Prompt Additions

When file descriptor system is enabled, the following instructions will be added to the enriched system prompt to explain the feature to the model:

```
<file_descriptor_instructions>
This system includes a file descriptor feature for handling large content:

1. When tool outputs or user inputs exceed character limits, they are automatically stored in file descriptors
2. Use the read_fd tool to access paginated content from file descriptors
3. File descriptors are referenced by their ID (e.g., "fd-12345")
4. You can read specific pages or the entire content at once

Key commands:
- read_fd(fd="fd-12345", page=2) - Read page 2
- read_fd(fd="fd-12345", read_all=True) - Read entire content

Important usage tips:
- Always read complete file descriptors before drawing conclusions
- Pay attention to the "truncated" and "continued" attributes that indicate content continuation
- For JSON data that spans multiple pages, consider reading all pages before parsing
- When working with very large content, consider using the fork tool to delegate analysis to child processes

Content formats include helpful XML metadata about pagination status and line continuations.
</file_descriptor_instructions>
```

## Implementation Plan

1. Basic file descriptor management (create, read)
2. Line-aware pagination 
3. Large user input detection and wrapping
4. Automatic wrapping for large tool outputs
5. System prompt enrichment with FD instructions
6. Integration with fork system call 
7. JSON pretty-printing support (optional)

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

6. **Batch Operations**: Support operations on multiple file descriptors at once
   ```python
   rename_fds({"fd-12345": "error_logs", "fd-67890": "config_file"})
   close_fds(["fd-12345", "fd-67890"])  # Close multiple FDs at once
   ```

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