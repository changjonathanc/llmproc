# RFC004: File Descriptor System Implementation Phases

This document outlines the implementation plan for the File Descriptor system, breaking down the work into distinct phases with clear milestones. For the complete system overview, see [RFC001: File Descriptor System for LLMProc](RFC001_file_descriptor_system.md).

## Implementation Status (As of 2025-03-25)

- ✅ **Phase 1**: Core Functionality - Completed
- 🔄 **Phase 2**: Process Integration - Partially Completed (Fork support implemented)
- 🔄 **Phase 2.5**: API Enhancements - Planned (See [RFC007](RFC007_fd_enhanced_api_design.md) for details)
- 🔄 **Phase 3**: Optional Features - Partially Completed (FD to File Operations implemented)

## Phase 1: Core Functionality _(Implemented)_

The initial phase focuses on essential functionality to establish the basic file descriptor system:

1. **Basic File Descriptor Manager** _(Implemented)_
   - In-memory storage of content
   - FD creation with unique IDs (sequential fd:1, fd:2, etc.)
   - Basic metadata tracking
   - Persistence as part of process state

2. **Read System Call** _(Implemented)_
   - Support for basic page-based reading
   - Support for reading entire content
   - Proper error handling for invalid FDs

3. **Line-Aware Pagination** _(Implemented)_
   - Index lines within content for efficient access
   - Break content at line boundaries when possible
   - Handle very long lines with character-based pagination
   - Track continuation status across pages

4. **Tool Output Wrapping** _(Implemented)_
   - Automatic detection of large tool outputs
   - Threshold-based FD creation
   - First page preview generation
   - XML formatting with consistent metadata

5. **XML Format Standards** _(Implemented)_
   - `<fd_result>` for wrapping large outputs
   - `<fd_content>` for read operation results
   - Standard attributes (fd, page, pages, continued, truncated)
   - Consistent metadata structure

6. **System Prompt Instructions** _(Implemented)_
   - Basic FD usage explanation
   - Read operation examples
   - Continuation handling guidance
   - Added automatically when FD system is enabled

## Phase 2: Process Integration

This phase focuses on integrating with the process model in two steps:

### Phase 2.1: Fork & Spawn Integration _(Partially Implemented)_

1. **Fork Integration** _(Implemented)_
   - Automatic FD copying during fork
   - Deep copy of all descriptors and metadata
   - Full inheritance of parent's FDs by child processes

2. **Cross-Process FD Access** _(Planned)_
   - Consistent FD referencing across forked / spawned processes
   - Inherited FDs remain accessible with same IDs
   - Support usage patterns with delegated reading (ask children to read a large fd)
   - Spawned processes inherit FDs from parent, but don't have knowledge of the fd, so the parent needs to provide the fd to the child

### Phase 2.2: Spawn Enhancements _(Planned)_

We need to implement the following:

1. **Spawn File Preloading**
   - Add `additional_preload_files` parameter to spawn
   - Support filesystem file preloading in child processes
   - Implement regardless of FD feature status
   - Update spawn tool schema and handler

2. **Spawn FD Preloading**
   - Add `additional_preload_fds` parameter for spawn
   - FD content inclusion in child enriched system prompt
   - Conditional schema based on FD feature status
   - Selective sharing of specific FDs with spawn

For detailed design, see [RFC005: File Descriptor Integration with Spawn Tool](RFC005_fd_spawn_integration.md).

## Phase 3: Optional Features

These features can be individually implemented and toggled:

1. **User Input Handling (Simple)** _(Planned)_
   - Detection of large user inputs
   - Automatic FD creation for inputs exceeding threshold
   - Preview generation with appropriate XML formatting
   - Configurable via `page_user_input` setting

2. **FD to File Operations** _(Implemented)_
   - Write FD content to filesystem files
   - Support for full content export
   - Parent directory creation if needed
   - Error handling with appropriate error types
   ```python
   # Example
   fd_to_file(fd="fd:12345", file_path="/path/to/output.txt")
   ```

3. **Enhanced FD API** _(Planned - See [RFC007](RFC007_fd_enhanced_api_design.md))_
   - Improved read_fd with create_fd parameter for content slicing
   - Enhanced fd_to_file with clear write/append modes
   - Explicit file creation control parameters
   ```python
   # Planned enhancements
   new_fd = read_fd(fd="fd:12345", page=2, create_fd=True)
   fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", mode="append")
   fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", fail_if_exists=True)
   ```
   
   _Note: The previous plan for page-specific export has been replaced with a more comprehensive API design in [RFC007](RFC007_fd_enhanced_api_design.md) that favors content slicing via read_fd with create_fd=True followed by fd_to_file operations._

4. **Response Reference ID System** _(Planned)_
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

   See [RFC006: Response Reference ID System](RFC006_response_reference_id.md) for detailed design.

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

5. **Selective Content Sharing**
   - Additional parameters for selectively preloading specific pages
   - Support for preloading specific line ranges
   - More precise control over what content is shared

## Optional Enhancements (Not Planned)

These features might be implemented if specific use cases emerge:

1. **Auto-Summarization**
   - LLM-based summarization of FD content
   - Quick overview of large content
   - Integration with FD metadata

2. **Section Referencing System**
   - Marking sections within file descriptors
   - Named references to specific content regions
   - Operations on marked sections

3. **Temporary Shared File System**
   - Temp files with kernel session persistence
   - Multi-process sharing of writable content
   - Integration with spawn and fork

4. **Search Capability**
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

## Current Implementation Location

The file descriptor system is implemented in the following files:

- `src/llmproc/tools/file_descriptor.py`: Core implementation of FileDescriptorManager and FD tools
- `src/llmproc/llm_process.py`: Integration with LLMProcess
- `src/llmproc/providers/anthropic_process_executor.py`: Integration with AnthropicProcessExecutor for output wrapping
- `src/llmproc/config/schema.py`: FileDescriptorConfig configuration schema
- `tests/test_file_descriptor.py`: Basic unit tests
- `tests/test_file_descriptor_integration.py`: Integration tests with process model
- `tests/test_fd_to_file_tool.py`: Tests for FD export functionality

The implementation follows the design outlined in this document, with a focus on modularity and clean interfaces. The FileDescriptorManager class is the central component responsible for FD creation, access, and pagination.

## Related RFCs

- [RFC001: File Descriptor System for LLMProc](RFC001_file_descriptor_system.md) - Main specification document
- [RFC003: File Descriptor Implementation Details](RFC003_file_descriptor_implementation.md) - Technical implementation details
- [RFC005: File Descriptor Integration with Spawn Tool](RFC005_fd_spawn_integration.md) - Integration with spawn system
- [RFC006: Response Reference ID System](RFC006_response_reference_id.md) - Integration with reference ID system
- [RFC007: Enhanced File Descriptor API Design](RFC007_fd_enhanced_api_design.md) - API improvements for read and write operations