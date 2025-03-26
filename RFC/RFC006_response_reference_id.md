# RFC006: Response Reference ID System

This document describes a feature that allows LLMs to mark sections of their responses with reference IDs, enabling easier reference and export of content. For the complete file descriptor system overview, see [RFC001: File Descriptor System for LLMProc](RFC001_file_descriptor_system.md).

## 1. Background

When an LLM generates a large or important piece of content (like code, analysis, or a long explanation), there's currently no way to:

1. Reference that content in subsequent interactions without repeating it
2. Export just that specific content to a file without manual copy-paste
3. Identify specific sections of a response for targeted operations

This contrasts with file descriptors, which focus on managing large *input* content (e.g., tool outputs). Response references complement this by handling *output* content, creating a unified system for content management.

## 2. Solution: Response Reference ID System

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

## 3. Key Features

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

## 4. API Design

### 4.1 Reference Tagging (LLM Output Format)

```xml
<ref id="example_code">
def hello_world():
    print("Hello, world!")
</ref>
```

### 4.2 Reference and File Descriptor Unified Interface

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

### 4.3 List Available References

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

### 4.4 Get Reference Content

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

## 5. Implementation Details

### 5.1 Reference Storage

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

### 5.2 Message Processing

The LLM output post-processor:

1. Scans messages for `<ref>` tags
2. Extracts and stores references in the process state
3. Leaves the content intact in the displayed message

### 5.3 System Prompt Additions

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

## 6. Integration with File Descriptors

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

## 7. Usage Scenarios

### 7.1 Code Generation

```
Human> Write a function to parse JSON from a file

Assistant> Here's a robust function for parsing JSON from a file:

<ref id="parse_json_function">
def parse_json_from_file(file_path):
    """Parse JSON data from a file with error handling.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data as Python objects
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    import json
    
    with open(file_path, 'r') as f:
        return json.load(f)
</ref>

You can use this function to safely parse JSON files.

Human> Save that to a utility file

Assistant> I'll save the JSON parsing function to a utility file for you.

fd_to_file(fd="ref:parse_json_function", file_path="utils/json_utils.py")

The function has been saved to utils/json_utils.py successfully.
```

### 7.2 Multiple References and Selective Export

```
Human> Write a complete Python module for working with CSV files

Assistant> Here's a complete CSV handling module:

<ref id="csv_module_imports">
import csv
import os
from typing import List, Dict, Any, Optional, Union, TextIO
</ref>

<ref id="csv_reader_function">
def read_csv(file_path: str, has_headers: bool = True) -> List[Dict[str, str]]:
    """Read a CSV file and return as a list of dictionaries.
    
    Args:
        file_path: Path to the CSV file
        has_headers: Whether the CSV has header row
        
    Returns:
        List of dictionaries (if has_headers=True) or list of lists
    """
    with open(file_path, 'r', newline='') as csvfile:
        if has_headers:
            reader = csv.DictReader(csvfile)
            return list(reader)
        else:
            reader = csv.reader(csvfile)
            return list(reader)
</ref>

<ref id="csv_writer_function">
def write_csv(data: List[Dict[str, Any]], file_path: str, fieldnames: Optional[List[str]] = None) -> None:
    """Write a list of dictionaries to a CSV file.
    
    Args:
        data: List of dictionaries to write
        file_path: Path to the output CSV file
        fieldnames: Optional list of field names for the CSV header
        
    Returns:
        None
    """
    if not data:
        return
        
    fieldnames = fieldnames or list(data[0].keys())
    
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
</ref>

<ref id="csv_module_full">
import csv
import os
from typing import List, Dict, Any, Optional, Union, TextIO

def read_csv(file_path: str, has_headers: bool = True) -> List[Dict[str, str]]:
    """Read a CSV file and return as a list of dictionaries.
    
    Args:
        file_path: Path to the CSV file
        has_headers: Whether the CSV has header row
        
    Returns:
        List of dictionaries (if has_headers=True) or list of lists
    """
    with open(file_path, 'r', newline='') as csvfile:
        if has_headers:
            reader = csv.DictReader(csvfile)
            return list(reader)
        else:
            reader = csv.reader(csvfile)
            return list(reader)

def write_csv(data: List[Dict[str, Any]], file_path: str, fieldnames: Optional[List[str]] = None) -> None:
    """Write a list of dictionaries to a CSV file.
    
    Args:
        data: List of dictionaries to write
        file_path: Path to the output CSV file
        fieldnames: Optional list of field names for the CSV header
        
    Returns:
        None
    """
    if not data:
        return
        
    fieldnames = fieldnames or list(data[0].keys())
    
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
</ref>

You can use these functions to easily read and write CSV files.

Human> Save just the reader function to csv_reader.py

Assistant> I'll save the CSV reader function to a separate file.

fd_to_file(fd="ref:csv_reader_function", file_path="csv_reader.py")

The CSV reader function has been saved to csv_reader.py successfully.

Human> Now save the complete module to csv_utils.py

Assistant> I'll save the complete CSV module to csv_utils.py.

fd_to_file(fd="ref:csv_module_full", file_path="csv_utils.py")

The complete CSV module has been saved to csv_utils.py successfully.
```

## 8. Implementation Plan

This feature will be implemented according to the timeline outlined in [RFC004: File Descriptor Implementation Phases](RFC004_fd_implementation_phases.md):

1. **Basic Functionality**
   - Reference extraction and storage
   - System prompt instructions
   - Integration with fd_to_file tool

2. **Enhanced Features**
   - list_refs and get_ref tool implementation
   - Integration with fork for reference inheritance
   - Advanced file operations (append, insert)

## 9. References

- [RFC001: File Descriptor System for LLMProc](RFC001_file_descriptor_system.md) - Main specification document
- [RFC003: File Descriptor Implementation Details](RFC003_file_descriptor_implementation.md) - Technical implementation details
- [RFC004: File Descriptor Implementation Phases](RFC004_fd_implementation_phases.md) - Implementation phases and status
- [RFC007: Enhanced File Descriptor API Design](RFC007_fd_enhanced_api_design.md) - Enhanced API features that will also benefit references