# RFC003: File Descriptor Implementation Details

This document focuses on the technical implementation details of the file descriptor system. For the complete system overview, architecture, and design principles, please see [RFC001: File Descriptor System for LLMProc](RFC001_file_descriptor_system.md).

## 1. Technical Implementation

### 1.1 FileDescriptorManager Class

The core of the implementation is the `FileDescriptorManager` class, which handles all aspects of file descriptor management:

```python
class FileDescriptorManager:
    """Manages file descriptors for large content.

    This class maintains a registry of active file descriptors, handling
    creation, reading, and pagination of content that exceeds context limits.

    Attributes:
        file_descriptors (dict): Dictionary mapping fd IDs to descriptor entries
        default_page_size (int): Default character count per page
        max_direct_output_chars (int): Threshold for FD creation
        fd_related_tools (set): Set of tool names that are part of the FD system
                              and should not trigger recursive FD creation
    """

    def __init__(
        self,
        default_page_size: int = 4000,
        max_direct_output_chars: int = 8000,
    ):
        """Initialize the FileDescriptorManager."""
        self.file_descriptors: Dict[str, Dict[str, Any]] = {}
        self.default_page_size = default_page_size
        self.max_direct_output_chars = max_direct_output_chars
        self.fd_related_tools = self._FD_RELATED_TOOLS.copy()
        self.next_fd_id = 1  # Counter for sequential FD IDs
    
    def create_fd(self, content: str, page_size: Optional[int] = None) -> Dict[str, Any]:
        """Create a new file descriptor for large content."""
        # Implementation details...
        
    def read_fd(self, fd_id: str, page: int = 1, read_all: bool = False) -> Dict[str, Any]:
        """Read content from a file descriptor."""
        # Implementation details...
        
    def write_fd_to_file(self, fd_id: str, file_path: str) -> Dict[str, Any]:
        """Write file descriptor content to a file."""
        # Implementation details...
        
    # Helper methods
    def _calculate_total_pages(self, fd_id: str) -> int:
        """Calculate the total number of pages in a file descriptor."""
        
    def _index_lines(self, content: str) -> Tuple[List[int], int]:
        """Create an index of line start positions."""
        
    def _get_page_content(self, fd_id: str, page: int) -> Tuple[str, Dict[str, Any]]:
        """Get content for a specific page with line-aware pagination."""
        
    def _format_fd_result(self, result: Dict[str, Any]) -> "ToolResult":
        """Format a file descriptor result in XML format."""
        
    def _format_fd_content(self, content: Dict[str, Any]) -> "ToolResult":
        """Format file descriptor content in XML format."""
        
    def _format_fd_file_result(self, result: Dict[str, Any]) -> "ToolResult":
        """Format file descriptor file operation result in XML format."""
        
    def _format_fd_error(self, error_type: str, fd_id: str, message: str) -> "ToolResult":
        """Format a file descriptor error in XML format."""
```

### 1.2 File Descriptor Internals

Each file descriptor is stored with the following structure:

```python
{
  "content": str,          # Full content
  "lines": list[int],      # Start indices of each line
  "total_lines": int,      # Total line count
  "page_size": int,        # Characters per page
  "creation_time": float,  # Creation timestamp
  "source": str,           # Origin of content (e.g., "tool_result", "user_input")
  "total_pages": int       # Total number of pages calculated from content
}
```

### 1.3 Line-Aware Pagination Implementation

The line-aware pagination algorithm follows these key steps:

```python
def _get_page_content(self, fd_id: str, page: int):
    """Get content for a specific page with line-aware pagination."""
    fd_entry = self.file_descriptors[fd_id]
    content = fd_entry["content"]
    page_size = fd_entry["page_size"]
    lines = fd_entry["lines"]
    
    # Calculate page boundaries
    start_char = (page - 1) * page_size
    end_char = min(start_char + page_size, len(content))
    
    # Find line boundaries for better pagination
    start_line = 1
    end_line = 1
    continued = False
    truncated = False
    
    # Find the start line (the line containing start_char)
    for i, line_start in enumerate(lines):
        if line_start > start_char:
            start_line = i  # The previous line
            break
        start_line = i + 1
    
    # Check if we're continuing from previous page
    if start_char > 0 and start_line > 1 and start_char != lines[start_line - 1]:
        continued = True
    
    # Find the end line (the line containing or after end_char)
    for i, line_start in enumerate(lines):
        if line_start >= end_char:
            end_line = i  # The previous line
            break
        end_line = i + 1
    
    # Check if we're truncating (not ending at line boundary)
    next_line_start = len(content)
    if end_line < len(lines):
        next_line_start = lines[end_line]
    
    if end_char < next_line_start:
        truncated = True
    
    # Extract the actual content
    page_content = content[start_char:end_char]
    
    return page_content, {
        "start_line": start_line,
        "end_line": end_line,
        "continued": continued,
        "truncated": truncated
    }
```

### 1.4 Tool Integration

The file descriptor system provides two main tools:

#### read_fd Tool

```python
async def read_fd_tool(
    fd: str,
    page: int = 1,
    read_all: bool = False,
    llm_process=None,
) -> "ToolResult":
    """Read content from a file descriptor."""
    from llmproc.tools.tool_result import ToolResult
    
    if not llm_process or not hasattr(llm_process, "fd_manager"):
        error_msg = "File descriptor operations require an LLMProcess with fd_manager"
        logger.error(f"READ_FD ERROR: {error_msg}")
        return ToolResult.from_error(error_msg)

    try:
        # Read from the file descriptor
        return llm_process.fd_manager.read_fd(fd, page=page, read_all=read_all)
    except Exception as e:
        error_msg = f"Error reading file descriptor: {str(e)}"
        logger.error(f"READ_FD ERROR: {error_msg}")
        logger.debug("Detailed traceback:", exc_info=True)
        return ToolResult.from_error(error_msg)
```

#### fd_to_file Tool

```python
async def fd_to_file_tool(
    fd: str,
    file_path: str,
    llm_process=None,
) -> "ToolResult":
    """Write file descriptor content to a file."""
    from llmproc.tools.tool_result import ToolResult
    
    if not llm_process or not hasattr(llm_process, "fd_manager"):
        error_msg = "File descriptor operations require an LLMProcess with fd_manager"
        logger.error(f"FD_TO_FILE ERROR: {error_msg}")
        return ToolResult.from_error(error_msg)
        
    try:
        # Write FD content to file
        return llm_process.fd_manager.write_fd_to_file(fd, file_path)
    except Exception as e:
        error_msg = f"Error writing file descriptor to file: {str(e)}"
        logger.error(f"FD_TO_FILE ERROR: {error_msg}")
        logger.debug("Detailed traceback:", exc_info=True)
        return ToolResult.from_error(error_msg)
```

### 1.5 Integration with LLMProcess

The file descriptor system is integrated with `LLMProcess` through several modifications:

```python
def __init__(self, program):
    # ... existing initialization ...
    
    # Initialize FD manager if file descriptor feature is enabled
    fd_config = getattr(program, "file_descriptor", None)
    if fd_config and fd_config.enabled:
        self.fd_manager = FileDescriptorManager(
            default_page_size=fd_config.default_page_size,
            max_direct_output_chars=fd_config.max_direct_output_chars
        )
        self.file_descriptor_enabled = True
    else:
        self.file_descriptor_enabled = False
        
    # ... rest of initialization ...
```

### 1.6 Tool Result Wrapping 

Large tool outputs are automatically wrapped into file descriptors:

```python
async def _process_tool_result(self, result, tool_name):
    # Skip FD wrapping for FD-related tools to avoid recursion
    if self.file_descriptor_enabled and not self.fd_manager.is_fd_related_tool(tool_name):
        content_str = str(result.content)
        
        # If result content exceeds threshold, store in FD and return FD info
        if len(content_str) > self.fd_manager.max_direct_output_chars:
            logger.debug(f"Wrapping large tool result ({len(content_str)} chars) in file descriptor")
            fd_info = self.fd_manager.create_fd(content_str)
            return fd_info  # This is already a ToolResult
            
    return result
```

### 1.7 XML Formatting

Each file descriptor operation uses consistent XML formatting:

```python
def _format_fd_result(self, result: Dict[str, Any]) -> "ToolResult":
    """Format a file descriptor result in XML format."""
    xml = (
        f'<fd_result fd="{result["fd"]}" pages="{result["pages"]}" '
        f'truncated="{str(result["truncated"]).lower()}" '
        f'lines="{result["lines"]}" total_lines="{result["total_lines"]}">\n'
        f'  <message>{result["message"]}</message>\n'
        f'  <preview>\n'
        f'  {result["preview"]}\n'
        f'  </preview>\n'
        f'</fd_result>'
    )
    
    return ToolResult(content=xml, is_error=False)
```

## 2. Performance Considerations

### 2.1 Memory Usage

For large outputs, memory usage is a critical concern:

- Each file descriptor stores the entire content in memory
- Line indexing requires additional storage proportional to the number of lines
- Multiple large file descriptors can lead to significant memory usage

Optimization techniques employed:

1. **Lazy pagination calculation**: Total pages are calculated only when needed and then cached
2. **Line indexing at creation**: Line boundaries are indexed once during FD creation
3. **Copy-on-fork semantics**: Child processes receive deep copies of FDs to prevent shared state issues

### 2.2 XML Response Formatting

XML formatting provides several advantages:

1. **Clear structure**: Separates metadata from content
2. **Consistent parsing**: LLMs can easily identify FD responses
3. **Extensibility**: New attributes can be added without breaking existing parsers
4. **Error handling**: Error responses follow the same XML pattern but with different tags

### 2.3 Optimized Line-Aware Pagination

The line-aware pagination algorithm provides these benefits:

1. **Preserves line boundaries**: Avoids cutting lines in the middle when possible
2. **Falls back gracefully**: Handles very long lines with character-based pagination
3. **Efficient calculation**: Uses pre-indexed line positions to avoid re-scanning content
4. **Clear continuation flags**: Explicitly marks when lines continue across pages

## 3. Security Considerations

To ensure the file descriptor system is secure:

1. **Access Control**: File descriptors are only accessible within the process that created them or inherited them
2. **Content Sanitization**: All content stored in FDs should be sanitized to prevent injection attacks
3. **Path Validation**: File paths for fd_to_file operations are validated to ensure they don't access sensitive areas
4. **Parent Directory Creation**: Parent directory creation is permitted to ease file operations but follows OS permissions
5. **Error Handling**: File operation errors are properly caught and reported without exposing system details

## 4. Code Organization

For the current implementation location and file structure, please refer to [RFC004: File Descriptor Implementation Phases](RFC004_fd_implementation_phases.md) which maintains the up-to-date implementation details.

## 5. Testing Approach

### 5.1 Unit Tests

Key unit tests include:

- Creation and reading of file descriptors with various contents
- Line-aware pagination with edge cases (long lines, empty content)
- Threshold testing (content sizes around the threshold values)
- Configuration validation and initialization

### 5.2 Integration Tests

Integration tests focus on:

- Automatic wrapping of large tool outputs
- System prompt integration
- FD tool registration and access
- Fork system integration
- Error handling with actual API calls

## 6. References

- [RFC001](RFC001_file_descriptor_system.md) - System Overview
- [RFC004](RFC004_fd_implementation_phases.md) - Implementation Phases
- [RFC005](RFC005_fd_spawn_integration.md) - Spawn Integration
- [RFC006](RFC006_response_reference_id.md) - Response Reference ID System
- [RFC007](RFC007_fd_enhanced_api_design.md) - Enhanced API Design