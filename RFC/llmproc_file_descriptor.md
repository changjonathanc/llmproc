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
3. **Close System Call** - For explicitly freeing resources when no longer needed

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

### 3. Close File Descriptor

```python
# Input
close_fd(fd="fd-12345")

# Output - XML-formatted response
"""
<fd_close fd="fd-12345" success="true">
  <message>File descriptor fd-12345 has been closed.</message>
</fd_close>
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
  "creation_time": str     # For potential future cleanup
}
```

### Integration with LLMProcess

1. Add FileDescriptorManager to LLMProcess state
2. Update fork_process to copy file descriptors
3. Add automatic wrapping of large tool outputs
4. Add automatic wrapping of large user inputs (stdin)
5. Register read_fd and close_fd tools
6. Add system prompt instructions about file descriptor usage

## Why Include close_fd?

Though not always necessary, close_fd is useful for:
1. Explicit resource management before fork/exec operations
2. Memory optimization in constrained environments
3. Maintaining the Unix-like file descriptor model

## Configuration

Simple TOML configuration:

```toml
[file_descriptor]
enabled = true                      # Enable file descriptor system
max_direct_output_chars = 8000      # Threshold for FD creation (larger than page size)
default_page_size = 4000            # Default page size for pagination
max_input_chars = 8000              # Threshold for creating FD from user input
json_pretty_print = true            # Pretty print JSON for better pagination (optional)
```

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

## Extension

- File descriptors will be shared and accessible when you `fork` a process
- This allows:
  - Main process can `fork` & delegate child processes to read existing file descriptors in detail
  - Parent process can fill fresh subprocesses with file descriptors as context

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
- close_fd(fd="fd-12345") - Close a file descriptor when no longer needed

Important usage tips:
- Always read complete file descriptors before drawing conclusions
- Pay attention to the "truncated" and "continued" attributes that indicate content continuation
- For JSON data that spans multiple pages, consider reading all pages before parsing
- When working with very large content, consider using the fork tool to delegate analysis to child processes

Content formats include helpful XML metadata about pagination status and line continuations.
</file_descriptor_instructions>
```

## Implementation Plan

1. Basic file descriptor management (create, read, close)
2. Line-aware pagination 
3. Large user input detection and wrapping
4. Automatic wrapping for large tool outputs
5. System prompt enrichment with FD instructions
6. Integration with fork system call 
7. JSON pretty-printing support (optional)