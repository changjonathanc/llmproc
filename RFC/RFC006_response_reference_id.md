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

1. **Simple Syntax with `<ref>` Tags**:
   - LLMs use familiar XML-style syntax: `<ref id="unique_id">content</ref>`
   - IDs are descriptive and unique within a conversation
   - Content is displayed normally to the user

2. **Seamless Integration with File Descriptor System**:
   - References use the existing file descriptor infrastructure
   - All file descriptor operations work with references
   - Unified access pattern via `ref:` namespace prefix

3. **Core Use Cases**:
   - Export specific sections of a response to files
   - Reference previously generated content without repetition
   - Process subsets of responses independently

## 4. Implementation Design

### 4.1 Simple Reference Syntax

```xml
<ref id="example_code">
def hello_world():
    print("Hello, world!")
</ref>
```

The syntax is:
- Familiar for LLMs (similar to HTML/XML)
- Clear to human readers
- Easy to parse programmatically
- Optional type attribute for specialized handling: `<ref id="hello" type="code">`

### 4.2 Unified Interface with File Descriptors

References integrate directly with the existing file descriptor system:

```python
# Export a reference to a file
fd_to_file(fd="ref:example_code", file_path="hello.py")

# Read a reference using standard FD tools
read_fd(fd="ref:example_code", read_all=True)
```

References use a namespaced identifier format:
- System-generated FDs: `fd:12345`
- LLM-generated references: `ref:example_code`

This prevents namespace collisions while providing a unified interface for all content operations.

## 5. Implementation Details

### 5.1 Reference Storage

References are automatically detected and stored in the file descriptor system:

```python
# When extracting references from LLM output:
reference_id = "example_code"
reference_content = "def hello_world():\n    print(\"Hello, world!\")"

# Store in the FD manager with reference namespace prefix
self.fd_manager.create_fd(
    content=reference_content, 
    source="reference",
    reference_id=reference_id
)
```

The system handles all the details:
- Automatically extracts references from LLM responses
- Creates file descriptors with proper namespacing
- Ensures content is displayed normally to the user

### 5.2 Reference Extraction

The implementation will:

1. Scan each assistant message for `<ref>` tags
2. Extract the ID and content from each reference
3. Store the content in the file descriptor system with the `ref:` prefix
4. Leave the original message unchanged for display

### 5.3 System Prompt Instructions

When enabled, these simple instructions are added to the system prompt:

```
<reference_instructions>
You can mark sections of your responses using reference tags:

<ref id="example_id">
Your content here (code, text, data, etc.)
</ref>

These references can be:
- Exported to files using: fd_to_file(fd="ref:example_id", file_path="output.txt")
- Read using standard file descriptor tools: read_fd(fd="ref:example_id", read_all=true)

Choose clear, descriptive IDs for your references.
</reference_instructions>
```

## 6. Integration with File Descriptor System

The reference ID system leverages the file descriptor infrastructure:

1. **Unified Storage**: References and file descriptors share the same backing store
2. **Consistent Tools**: All FD tools work with references using the `ref:` prefix
3. **Simple Implementation**: No need for a separate system just for references

Benefits of this integration:

- **Efficiency**: Reuses existing pagination and storage mechanisms
- **Simplicity**: LLMs only need to learn one basic tag format
- **Flexibility**: All file descriptor operations work with references

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

## 8. Implementation Steps

The reference ID system will be implemented in two simple phases:

### Phase 1: Core Functionality
1. Add reference extraction in the LLMProcess._async_run method
2. Integrate with the file descriptor system for storage
3. Update system prompt with reference instructions
4. Test with basic use cases (creating and exporting references)

### Phase 2: Refinements
1. Ensure references are properly preserved during fork operations
2. Add reference tracking to display created references
3. Fine-tune the format of referenced content in file descriptor system

## 9. References

- [RFC001: File Descriptor System for LLMProc](RFC001_file_descriptor_system.md) - Main specification document
- [RFC004: File Descriptor Implementation Phases](RFC004_fd_implementation_phases.md) - Implementation phases and status