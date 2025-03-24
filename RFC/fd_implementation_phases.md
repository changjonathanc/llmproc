# File Descriptor System Implementation Phases

This document outlines the implementation plan for the File Descriptor system, breaking down the work into distinct phases with clear milestones.

## Phase 1: Core Functionality

The initial phase focuses on essential functionality to establish the basic file descriptor system:

1. **Basic File Descriptor Manager**
   - In-memory storage of content
   - FD creation with unique IDs
   - Basic metadata tracking
   - Persistence as part of process state

2. **Read System Call**
   - Support for basic page-based reading
   - Support for reading entire content
   - Proper error handling for invalid FDs

3. **Line-Aware Pagination**
   - Index lines within content for efficient access
   - Break content at line boundaries when possible
   - Handle very long lines with character-based pagination
   - Track continuation status across pages

4. **Tool Output Wrapping**
   - Automatic detection of large tool outputs
   - Threshold-based FD creation
   - First page preview generation
   - XML formatting with consistent metadata

5. **XML Format Standards**
   - `<fd_result>` for wrapping large outputs
   - `<fd_content>` for read operation results
   - Standard attributes (fd, page, pages, continued, truncated)
   - Consistent metadata structure

6. **System Prompt Instructions**
   - Basic FD usage explanation
   - Read operation examples
   - Continuation handling guidance
   - Added automatically when FD system is enabled

## Phase 2: Process Integration

This phase focuses on integrating file descriptors with the process model:

1. **Fork Integration**
   - Automatic FD copying during fork
   - Deep copy of all descriptors and metadata
   - Full inheritance of parent's FDs by child processes

2. **Cross-Process FD Access**
   - Consistent FD referencing across forked processes
   - Inherited FDs remain accessible with same IDs
   - Support usage patterns with delegated reading

3. **Spawn Integration**
   - `additional_preload_fds` parameter for spawn
   - FD content inclusion in child enriched system prompt
   - Selective sharing of specific FDs with spawn

## Phase 3: Optional Features

These features can be individually implemented and toggled:

1. **User Input Handling (Simple)**
   - Detection of large user inputs
   - Automatic FD creation for inputs exceeding threshold
   - Preview generation with appropriate XML formatting
   - Configurable via `page_user_input` setting

2. **FD to File Operations**
   - Write FD content to filesystem files
   - Support for full content export
   - Support for page-specific export
   - Support for line range export
   - File operation modes (write, append, insert)
   ```python
   # Examples
   fd_to_file(fd="fd-12345", file_path="/path/to/output.txt", mode="write")
   fd_to_file(fd="fd-12345", file_path="/path/to/output.txt", page=2, mode="write")
   fd_to_file(fd="fd-12345", file_path="/path/to/output.txt", start_line=45, end_line=90, mode="append")
   ```

3. **Selective Content Sharing**
   - Additional parameters for selectively preloading specific pages
   - Support for preloading specific line ranges
   - More precise control over what content is shared

## Future Work

Features planned for later development:

1. **Disk Checkpointing**
   - Combined checkpoint system for state and FDs
   - Process hibernation support
   - Crash recovery with complete context
   ```python
   checkpoint_id = checkpoint_process()
   restore_process(checkpoint_id)
   ```

2. **Universal Conversation FD References**
   - Automatic FD assignment to conversation messages
   - Referenceable history through FD system
   - Integration with preloading for cross-process sharing

3. **Chunk-Aware User Input Processing**
   - Support for multi-part user messages
   - Selective pagination of only large chunks
   - Preservation of message structure

4. **Semantic Navigation**
   - Support for navigating by units like paragraphs
   - Code-aware navigation (by function, class)
   - More intuitive content access patterns

## Additional Features

These features complement the file descriptor system and address different use cases:

1. **Response Reference ID System**
   - Allow LLMs to mark sections of their responses with reference IDs
   - Enable exporting referenced content to files
   - Reference specific parts of previous responses
   - File operations on referenced content
   
   ```python
   # Example response with reference ID
   <ref id="fibonacci_code">
   def fibonacci(n):
       a, b = 0, 1
       for _ in range(n):
           a, b = b, a + b
       return a
   </ref>
   
   # Export to file using standard fd_to_file
   fd_to_file(fd="ref:fibonacci_code", file_path="fibonacci.py")
   ```
   
   See `response_reference_id.md` for detailed design.

## Optional Enhancements (Not Planned)

These features might be implemented if specific use cases emerge:

1. **JSON Pretty Printing**
   - Automatic detection of JSON content
   - Reformatting for better readability
   - Improved pagination with structured format

2. **Auto-Summarization**
   - LLM-based summarization of FD content
   - Quick overview of large content
   - Integration with FD metadata

3. **Section Referencing System**
   - Marking sections within file descriptors
   - Named references to specific content regions
   - Operations on marked sections

4. **Temporary Shared File System**
   - Temp files with kernel session persistence
   - Multi-process sharing of writable content
   - Integration with spawn and fork

5. **Search Capability**
   - Pattern matching within file descriptors
   - Content-aware searching
   - Integration with section markers

## Note on Deliberate Omissions

Features deliberately not included in the implementation plan:

1. **close_fd System Call**
   - Explicitly omitted to treat FDs as persistent state
   - Simplifies inheritance and cross-process behavior
   - Avoids issues with dangling references
   - Resource management handled at process termination
   - Future disk offloading will address memory constraints

2. **Batch Operations on FDs**
   - Without close_fd, most batch operations aren't needed
   - Individual operations are sufficient for core functionality