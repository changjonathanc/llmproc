# Response Reference ID System

This document describes a feature that allows LLMs to mark sections of their responses with reference IDs, enabling easier reference and export of content.

## Problem

When an LLM generates a large or important piece of content (like code, analysis, or a long explanation), there's currently no way to:

1. Reference that content in subsequent interactions without repeating it
2. Export just that specific content to a file without manual copy-paste
3. Identify specific sections of a response for targeted operations

## Solution: Response Reference ID System

Allow LLMs to mark sections of their responses with unique reference IDs that can be used later:

```
Human> Write a Python function to calculate Fibonacci numbers

Assistant> Here's a recursive implementation:

<ref id="fib_recursive">
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
</ref>

And here's a more efficient iterative version:

<ref id="fib_iterative">
def fibonacci_iterative(n):
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b
</ref>

You can use either version depending on your needs.

Human> Export the iterative version to a file

Assistant> I'll export the iterative version for you.

fd_to_file(fd="ref:fib_iterative", file_path="fibonacci.py")

Done! The iterative Fibonacci function has been saved to fibonacci.py.
```

## Key Features

1. **Response Marking with `<ref>` Tags**:
   - LLMs can wrap content in `<ref id="unique_id">content</ref>` tags
   - IDs are unique within a conversation
   - Content is displayed normally to the user

2. **Reference System**:
   - References persist in the conversation state
   - Can be accessed by the LLM in subsequent turns
   - Are fully copied during fork operations

3. **File Operations with References**:
   - Export referenced content to files
   - Append referenced content to existing files
   - Insert referenced content at specific points in files

## API Design

### 1. Reference Tagging (LLM Output Format)

```xml
<ref id="example_code">
def hello_world():
    print("Hello, world!")
</ref>
```

### 2. Reference and File Descriptor Unified Interface

References are automatically stored as file descriptors with the reference ID serving as the descriptor ID:

```python
# Use references directly with fd_to_file
fd_to_file(fd="ref:example_code", file_path="/path/to/hello.py", mode="write")

# Use any file descriptor operation with references
read_fd(fd="ref:example_code", page=1)
```

References use a namespaced identifier format:
- System-generated FDs: `fd:12345`
- LLM-generated references: `ref:example_code`

This prevents namespace collisions while providing a unified interface for all content operations. LLMs can use the same tools for both system-generated FDs and their own references.

### 3. List Available References

```python
# List all references in the conversation
list_refs()

# Output - XML-formatted response
"""
<ref_list count="3">
  <ref id="example_code" created="2025-03-24T14:30:00" lines="3" chars="48" />
  <ref id="fibonacci" created="2025-03-24T14:35:00" lines="10" chars="156" />
  <ref id="data_analysis" created="2025-03-24T14:40:00" lines="25" chars="820" />
</ref_list>
"""
```

### 4. Get Reference Content

Getting reference content uses the file descriptor system under the hood:

```python
# Get a reference's content for reuse
get_ref(ref_id="example_code")

# Output - XML-formatted response
"""
<ref_content id="example_code">
def hello_world():
    print("Hello, world!")
</ref_content>
"""
```

This operation internally:
1. Gets the file descriptor associated with the reference
2. Uses read_fd with read_all=True to retrieve content
3. Formats the response with reference-specific metadata

## Implementation Details

### Reference Storage

References are stored directly in the file descriptor system:

```python
# When extracting references from LLM output, the system:
reference_id = "example_code"
reference_content = "def hello_world():\n    print(\"Hello, world!\")"

# Creates an FD with the reference namespace prefix
fd_id = f"ref:{reference_id}"  # "ref:example_code"

# Stores in the FD manager alongside system-generated FDs
self.fd_manager.create_fd(fd_id, content=reference_content)
```

This integration avoids duplicate storage mechanisms and makes all content management functions available for references.

### Message Processing

The LLM output post-processor:

1. Scans messages for `<ref>` tags
2. Extracts and stores references in the process state
3. Leaves the content intact in the displayed message

### System Prompt Additions

When reference ID system is enabled, these instructions are added to the system prompt:

```
<reference_id_instructions>
This system includes a reference ID feature for marking and referencing parts of your responses:

1. You can mark important parts of your response with <ref id="unique_id">content</ref> tags
2. These references can later be exported to files or reused without repetition
3. Choose descriptive IDs for your references (e.g., "recursive_search_function")
4. References persist throughout the conversation

Key commands:
- fd_to_file(fd="ref:example_code", file_path="hello.py") - Write to file
- list_refs() - List all available references
- read_fd(fd="ref:example_code", read_all=True) - Read reference content

This feature is particularly useful for code, structured data, or any content that might
need to be exported to a file or referenced later.
</reference_id_instructions>
```

## Integration with File Descriptors

The reference ID system is tightly integrated with the file descriptor system:

1. **File Descriptors**: Handle large **input** content (tool outputs, user input)
2. **Reference IDs**: Handle specific parts of **output** content (LLM responses)
3. **Unified Storage**: References are stored as file descriptors internally

When a reference is created, the system:
1. Creates a file descriptor with the referenced content
2. Maintains a mapping between reference IDs and their corresponding FD IDs
3. Uses the existing FD infrastructure for content storage and retrieval

This provides:
- Unified storage and pagination system for both input and output content
- Consistent APIs for working with large content
- Reuse of file operation tools (fd_to_file) for references
- Resource optimization through shared infrastructure

## Implementation Plan

1. **Basic Functionality**
   - Reference extraction and storage
   - System prompt instructions
   - ref_to_file tool implementation

2. **Enhanced Features**
   - List and get reference tools
   - Integration with fork for reference inheritance
   - Advanced file operations (append, insert)

3. **Extensions**
   - Auto-suggestion of reference IDs for code blocks
   - Integration with file descriptors for efficient storage
   - Reference categorization (code, data, text, etc.)