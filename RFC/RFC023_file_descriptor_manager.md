# RFC023: FileDescriptorManager Extraction

## Overview
This RFC proposes extracting all file descriptor-related functionality from `LLMProcess` into a dedicated `FileDescriptorManager` class to improve code organization, maintainability, and enable better separation of concerns.

## Motivation
The current implementation of file descriptor functionality is embedded throughout the `LLMProcess` class, making it difficult to understand, test, and maintain. By extracting this functionality into a dedicated class, we can:

1. Reduce complexity in the `LLMProcess` class
2. Establish clear boundaries between FD operations and core LLM functionality
3. Improve testability with focused unit tests
4. Reduce the file size of `llm_process.py` (currently ~657 lines)
5. Make both components more maintainable

## Implementation Details

### New File Structure
- Create a new file: `/src/llmproc/file_descriptors/fd_manager.py`
- Create a new directory if needed: `/src/llmproc/file_descriptors/`

### FileDescriptorManager Class
```python
class FileDescriptorManager:
    """Manages file descriptors for large content handling.
    
    This class handles creation, reading, and management of file descriptors,
    including pagination, extraction, and reference tracking.
    """
    
    def __init__(self, llm_process=None):
        """Initialize the file descriptor manager.
        
        Args:
            llm_process: The parent LLMProcess instance (for callbacks)
        """
        self.llm_process = llm_process
        self.fds = {}  # Maps FD IDs to content
        self.fd_metadata = {}  # Stores metadata about FDs (source, etc.)
        self.reference_counter = 0  # For generating reference IDs
        self.fd_tool_names = set(["read_fd", "fd_to_file"])  # Default FD tools
        self.seen_references = set()  # Track processed references
        self.fd_counter = 0  # For generating new FD IDs
```

### Methods to Extract from LLMProcess

The following methods should be moved from `LLMProcess` to `FileDescriptorManager`:

1. **FD Creation and Management**
   ```python
   def create_fd(self, content, metadata=None):
       """Create a new file descriptor with content."""
       # Existing implementation...
       
   def get_fd(self, fd_id):
       """Get content of a file descriptor."""
       # Existing implementation...
       
   def delete_fd(self, fd_id):
       """Delete a file descriptor."""
       # Existing implementation...
       
   def list_fds(self):
       """List all file descriptors."""
       # Existing implementation...
       
   def reset_fds(self):
       """Reset all file descriptors."""
       # Existing implementation...
   ```

2. **Pagination and Reading**
   ```python
   def paginate_fd(self, fd_id, mode="page", start=1, count=1):
       """Paginate content from a file descriptor."""
       # Existing implementation...
       
   def get_fd_page_count(self, fd_id, mode="page"):
       """Get the number of pages in a file descriptor."""
       # Existing implementation...
       
   def extract_portion(self, fd_id, start_line, end_line):
       """Extract a portion of content from a file descriptor."""
       # Existing implementation...
   ```

3. **Reference Processing**
   ```python
   def process_references(self, message):
       """Process references in a message."""
       # Existing implementation...
       
   def extract_reference_ids(self, message):
       """Extract reference IDs from a message."""
       # Existing implementation...
       
   def generate_reference_id(self):
       """Generate a new reference ID."""
       # Existing implementation...
       
   def is_reference_seen(self, ref_id):
       """Check if a reference has been seen."""
       # Existing implementation...
       
   def mark_reference_seen(self, ref_id):
       """Mark a reference as seen."""
       # Existing implementation...
   ```

4. **Integration Points**
   ```python
   def register_fd_tool(self, tool_name):
       """Register a tool as FD-related."""
       # Existing implementation...
       
   def is_fd_related_tool(self, tool_name):
       """Check if a tool is FD-related."""
       # Existing implementation...
       
   def process_fd_in_user_input(self, user_input):
       """Process file descriptors in user input."""
       # Existing implementation...
   ```

### Changes to LLMProcess

The `LLMProcess` class will need to be updated to delegate to the new `FileDescriptorManager`:

```python
class LLMProcess:
    def __init__(self, program=None, **kwargs):
        # Existing initialization...
        
        # Initialize FD manager
        from llmproc.file_descriptors.fd_manager import FileDescriptorManager
        self.fd_manager = FileDescriptorManager(self)
        
        # For backward compatibility
        self.file_descriptor_enabled = True if program.file_descriptor_enabled else False
        
    # Delegation methods for backward compatibility
    def create_fd(self, content, metadata=None):
        return self.fd_manager.create_fd(content, metadata)
        
    def get_fd(self, fd_id):
        return self.fd_manager.get_fd(fd_id)
        
    # Additional delegation methods for all FD functions...
```

### Migration Strategy
1. Create the new `FileDescriptorManager` class with all extracted methods
2. Update `LLMProcess` to use the new class while maintaining backward compatibility
3. Add appropriate tests for the new class
4. Update any import statements in affected files

## Benefits
1. **Reduced Complexity**: `LLMProcess` becomes more focused on its core responsibilities
2. **Improved Testability**: Easier to write unit tests for FD functionality
3. **Better Maintainability**: Changes to the FD system can be made without modifying `LLMProcess`
4. **Clearer API**: Explicit boundaries between different subsystems
5. **Reduced File Size**: Both `llm_process.py` and the extracted file are under 500 lines

## Backward Compatibility
All existing methods in `LLMProcess` related to file descriptors will be maintained but will delegate to the new `FileDescriptorManager`. This ensures no breaking changes for external code.

## Future Work
Future enhancements could include:
1. Enhanced file descriptor types (e.g., specialized for code, spreadsheets, etc.)
2. Improved pagination algorithms for different content types
3. Persistent storage of file descriptors
4. Statistics collection for file descriptor usage
5. More sophisticated reference tracking mechanisms