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

ref_to_file(ref_id="fib_iterative", file_path="fibonacci.py")

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

### 2. Reference to File Operation

```python
# Write reference content to a file
ref_to_file(ref_id="example_code", file_path="/path/to/hello.py", mode="write")

# Append to an existing file
ref_to_file(ref_id="example_code", file_path="/path/to/examples.py", mode="append")

# Insert at a specific line
ref_to_file(ref_id="example_code", file_path="/path/to/examples.py", insert_at_line=10, mode="insert")

# Output - XML-formatted response
"""
<ref_write ref_id="example_code" file_path="/path/to/hello.py" success="true" mode="write">
  <message>Reference content successfully written to /path/to/hello.py</message>
  <stats>
    <bytes>48</bytes>
    <lines>3</lines>
  </stats>
</ref_write>
"""
```

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

## Implementation Details

### Reference Storage

References are stored in the LLMProcess state:

```python
self.references = {
    "example_code": {
        "content": "def hello_world():\n    print(\"Hello, world!\")",
        "created": "2025-03-24T14:30:00",
        "lines": 3,
        "chars": 48
    },
    # ...
}
```

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
- ref_to_file(ref_id="example_code", file_path="hello.py") - Write to file
- list_refs() - List all available references
- get_ref(ref_id="example_code") - Get reference content

This feature is particularly useful for code, structured data, or any content that might
need to be exported to a file or referenced later.
</reference_id_instructions>
```

## Integration with File Descriptors

The reference ID system complements the file descriptor system:

1. **File Descriptors**: Handle large **input** content (tool outputs, user input)
2. **Reference IDs**: Handle specific parts of **output** content (LLM responses)

Together, they provide a comprehensive system for managing large content in both directions.

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