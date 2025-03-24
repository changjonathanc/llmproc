"""File descriptor system call for LLMProcess to handle large tool outputs."""

import logging
import uuid
from math import ceil
from typing import Any, Dict, Optional

from llmproc.tools.tool_result import ToolResult

# Set up logger
logger = logging.getLogger(__name__)

# Default max output length before pagination is used
DEFAULT_MAX_DIRECT_OUTPUT_TOKENS = 1000
# Default page size for paginated outputs
DEFAULT_PAGE_SIZE = 1000

# Detailed read_fd tool description
read_fd_tool_description = """
This tool lets you read content from a file descriptor (fd) when output from other tools exceeds
the token limit and is stored in the process memory.

When a tool's result is very large, instead of returning the entire content, the system will:
1. Store the content and assign it a file descriptor (fd)
2. Return a truncated version with the first page and metadata
3. Allow you to access additional content using this read_fd tool

Usage:
read_fd(fd, page=1)

Parameters:
- fd: The file descriptor ID returned from a previous tool call
- page: Page number to read (starting from 1, default=1)

Example:
When a search returns too much content, you'll get:
{
  "fd": "fd-12345",
  "message": "Output exceeds 1000 tokens, use read_fd to read more",
  "total_length": 5432,
  "num_pages": 6,
  "output_preview": "First page content..."
}

Then you can read specific pages:
read_fd(fd="fd-12345", page=2)  # Read page 2

Tips:
- Always check if a tool response contains an "fd" field
- Read all needed pages before drawing conclusions
- Particularly useful when working with long text outputs from searches, file reads, etc.
"""

# Definition of the read_fd tool for API
read_fd_tool_def = {
    "name": "read_fd",
    "description": read_fd_tool_description,
    "input_schema": {
        "type": "object",
        "properties": {
            "fd": {
                "type": "string",
                "description": "File descriptor ID to read from"
            },
            "page": {
                "type": "integer",
                "description": "Page number to read (starting from 1)",
                "default": 1
            }
        },
        "required": ["fd"]
    },
}


class FileDescriptorManager:
    """Manages file descriptors for LLMProcess.
    
    This class provides methods to create, read, and manage file descriptors
    for large tool outputs that exceed token limits.
    """
    
    def __init__(self):
        """Initialize an empty file descriptor manager."""
        self.file_descriptors = {}
        
    def create_fd(self, content: str, max_direct_output: int = DEFAULT_MAX_DIRECT_OUTPUT_TOKENS) -> Dict[str, Any]:
        """Create a new file descriptor for content that exceeds the token limit.
        
        Args:
            content: The full content to store
            max_direct_output: Maximum tokens to include in the preview
            
        Returns:
            A dictionary with fd info including preview content
        """
        # Generate a unique file descriptor ID
        fd_id = f"fd-{uuid.uuid4().hex[:8]}"
        
        # Calculate number of pages (approximate token estimation)
        total_length = len(content)
        num_pages = ceil(total_length / DEFAULT_PAGE_SIZE)
        
        # Get first page for preview
        preview = content[:max_direct_output] if content else ""
        if len(content) > max_direct_output:
            preview += "..."
            
        # Store the content with metadata
        self.file_descriptors[fd_id] = {
            "content": content,
            "total_length": total_length,
            "num_pages": num_pages,
            "page_size": DEFAULT_PAGE_SIZE,
            "creation_time": uuid.uuid1().hex  # For potential cleanup later
        }
        
        # Prepare the response
        fd_info = {
            "fd": fd_id,
            "message": f"Output exceeds {max_direct_output} tokens, use read_fd to read more",
            "total_length": total_length,
            "num_pages": num_pages,
            "output_preview": preview
        }
        
        logger.debug(f"Created file descriptor {fd_id} with {num_pages} pages")
        return fd_info
    
    def read_fd(self, fd: str, page: int = 1) -> Optional[Dict[str, Any]]:
        """Read a specific page from a file descriptor.
        
        Args:
            fd: The file descriptor ID
            page: Page number to read (starting from 1)
            
        Returns:
            A dictionary with page content and metadata, or None if fd not found
        """
        # Check if the file descriptor exists
        if fd not in self.file_descriptors:
            return None
            
        fd_data = self.file_descriptors[fd]
        
        # Validate page number
        if page < 1 or page > fd_data["num_pages"]:
            return {
                "error": f"Invalid page number. Valid range: 1-{fd_data['num_pages']}",
                "fd": fd,
                "num_pages": fd_data["num_pages"]
            }
            
        # Calculate page boundaries
        start_idx = (page - 1) * fd_data["page_size"]
        end_idx = min(start_idx + fd_data["page_size"], fd_data["total_length"])
        
        # Extract the page content
        content = fd_data["content"][start_idx:end_idx]
        
        # Prepare the response
        result = {
            "fd": fd,
            "page": page,
            "num_pages": fd_data["num_pages"],
            "content": content,
            "page_range": f"{start_idx+1}-{end_idx} of {fd_data['total_length']}"
        }
        
        logger.debug(f"Read page {page} from file descriptor {fd}")
        return result
    
    def close_fd(self, fd: str) -> bool:
        """Close and remove a file descriptor.
        
        Args:
            fd: The file descriptor ID to close
            
        Returns:
            True if closed successfully, False if fd not found
        """
        if fd in self.file_descriptors:
            del self.file_descriptors[fd]
            logger.debug(f"Closed file descriptor {fd}")
            return True
        return False


async def read_fd_tool(fd: str, page: int = 1, llm_process=None) -> ToolResult:
    """Read content from a file descriptor stored in the process.
    
    Args:
        fd: The file descriptor ID to read from
        page: Page number to read (starting from 1, default=1)
        llm_process: The LLMProcess instance that owns the file descriptors
        
    Returns:
        ToolResult with content or error message
    """
    if not llm_process:
        return ToolResult.from_error("File descriptor operation failed: No process context available")
    
    # Ensure the process has a file descriptor manager
    if not hasattr(llm_process, "fd_manager"):
        return ToolResult.from_error("File descriptor operation failed: File descriptor system not initialized")
    
    # Read from the file descriptor
    result = llm_process.fd_manager.read_fd(fd, page)
    
    if result is None:
        return ToolResult.from_error(f"File descriptor not found: {fd}")
        
    if "error" in result:
        return ToolResult.from_error(result["error"])
        
    return ToolResult.from_success(result)