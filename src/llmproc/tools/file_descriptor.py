"""File descriptor system for managing large tool outputs.

This module implements a file descriptor (FD) system that handles the storage,
pagination, and access of large content that exceeds context limits. It provides
a Unix-like file descriptor abstraction for LLM processes with standard tools
for reading paginated content.

The system includes:
1. FileDescriptorManager - Core class managing FD creation and access
2. read_fd tool - Standard interface for accessing paginated content
3. fd_to_file tool - Export file descriptor content to a file
4. Support functions for line-aware pagination and FD formatting
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

from llmproc.tools.tool_result import ToolResult

# Set up logger
logger = logging.getLogger(__name__)

# Tool descriptions
read_fd_tool_description = """
Reads content from a file descriptor that was created to store large tool outputs.
When tool outputs exceed the context limit, they are stored in a file descriptor,
and you'll need to use this tool to read the content in pages.

Usage:
  read_fd(fd="fd:12345", page=2) - Read page 2 from the file descriptor
  read_fd(fd="fd:12345", read_all=True) - Read the entire content (use cautiously with very large content)
  read_fd(fd="fd:12345", page=2, extract_to_new_fd=True) - Create a new FD containing just page 2
  
Parameters:
  fd (str): The file descriptor ID to read from (e.g., "fd:12345")
  page (int, optional): The page number to read (starting from 1)
  read_all (bool, optional): If true, returns the entire content (may be very large)
  extract_to_new_fd (bool, optional): If true, extracts the content to a new file descriptor and returns the new FD ID

When to use this tool:
- When you see a file descriptor reference (fd:12345) in a tool result
- When you need to read more pages from large content
- When you want to analyze content that was too large to include directly in the response
- When you need to extract a specific part of a large content into a new FD (use extract_to_new_fd=True)
"""

fd_to_file_tool_description = """
Writes the content of a file descriptor to a file on disk.
This tool is useful when you want to save large content for later use.

Usage:
  fd_to_file(fd="fd:12345", file_path="/path/to/output.txt") - Write content to a file (create or overwrite)
  fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", mode="append") - Append content to existing file
  fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", create=False) - Write only if file exists
  fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", exist_ok=False) - Create only if file doesn't exist
  
Parameters:
  fd (str): The file descriptor ID to export (e.g., "fd:12345")
  file_path (str): The path to the file to write
  mode (str, optional): "write" (default) or "append" - Whether to overwrite or append to file
  create (bool, optional): True (default) or False - Whether to create the file if it doesn't exist
  exist_ok (bool, optional): True (default) or False - Whether it's ok if the file already exists

Behavior matrix:
  - mode="write", create=True, exist_ok=True: Create or overwrite (default)
  - mode="write", create=True, exist_ok=False: Create only if doesn't exist
  - mode="write", create=False, exist_ok=True: Update existing only
  - mode="append", create=True, exist_ok=True: Append, create if needed
  - mode="append", create=True, exist_ok=False: Append only if exists, else create new
  - mode="append", create=False, exist_ok=True: Append to existing only

When to use this tool:
- When you need to save file descriptor content to disk
- When you want to process the content with other tools that work with files
- When you need to preserve large output for later reference
- When you need to append content to existing files (use mode="append")
- When you need precise control over file creation and overwriting behavior
"""

# Tool definitions for Anthropic API
read_fd_tool_def = {
    "name": "read_fd",
    "description": read_fd_tool_description,
    "input_schema": {
        "type": "object",
        "properties": {
            "fd": {
                "type": "string",
                "description": "The file descriptor ID to read from (e.g., 'fd:12345')",
            },
            "page": {
                "type": "integer",
                "description": "The page number to read (starting from 1)",
            },
            "read_all": {
                "type": "boolean",
                "description": "If true, returns the entire content (use cautiously with very large content)",
            },
            "extract_to_new_fd": {
                "type": "boolean",
                "description": "If true, extracts the content to a new file descriptor and returns the new FD ID",
            },
        },
        "required": ["fd"],
    },
}

fd_to_file_tool_def = {
    "name": "fd_to_file",
    "description": fd_to_file_tool_description,
    "input_schema": {
        "type": "object",
        "properties": {
            "fd": {
                "type": "string",
                "description": "The file descriptor ID to export (e.g., 'fd:12345')",
            },
            "file_path": {
                "type": "string",
                "description": "The path to the file to write",
            },
            "mode": {
                "type": "string",
                "enum": ["write", "append"],
                "description": "Whether to overwrite ('write', default) or append ('append') to the file",
            },
            "create": {
                "type": "boolean",
                "description": "Whether to create the file if it doesn't exist (default: true)",
            },
            "exist_ok": {
                "type": "boolean",
                "description": "Whether it's ok if the file already exists (default: true)",
            },
        },
        "required": ["fd", "file_path"],
    },
}

# System prompt instructions for file descriptor usage
file_descriptor_instructions = """
<file_descriptor_instructions>
This system includes a file descriptor feature for handling large content:

1. Large outputs are stored in file descriptors (fd:12345)
2. Use read_fd to access content in pages or all at once
3. Use fd_to_file to export content to disk files

Key commands:
- read_fd(fd="fd:12345", page=2) - Read page 2
- read_fd(fd="fd:12345", read_all=True) - Read entire content
- read_fd(fd="fd:12345", extract_to_new_fd=True) - Extract content to a new FD
- fd_to_file(fd="fd:12345", file_path="/path/to/output.txt") - Save to file
- fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", mode="append") - Append to file
- fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", exist_ok=False) - Create new file only
- fd_to_file(fd="fd:12345", file_path="/path/to/output.txt", create=False) - Update existing file only

Tips:
- Check "truncated" and "continued" attributes for content continuation
- When analyzing large content, consider reading all pages first
- Use extract_to_new_fd=True when you need to extract specific content
- Use mode="append" when adding to existing files
- Use exist_ok=False to avoid overwriting existing files
- Use create=False when you want to update only existing files
</file_descriptor_instructions>
"""


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

    # Registry of FD-related tools that should not trigger recursive FD creation
    _FD_RELATED_TOOLS = {"read_fd", "fd_to_file"}
    
    def __init__(
        self,
        default_page_size: int = 4000,
        max_direct_output_chars: int = 8000,
    ):
        """Initialize the FileDescriptorManager.

        Args:
            default_page_size: Default number of characters per page
            max_direct_output_chars: Threshold for automatic FD creation
        """
        self.file_descriptors: Dict[str, Dict[str, Any]] = {}
        self.default_page_size = default_page_size
        self.max_direct_output_chars = max_direct_output_chars
        self.fd_related_tools = self._FD_RELATED_TOOLS.copy()
        self.next_fd_id = 1  # Counter for sequential FD IDs
    
    def is_fd_related_tool(self, tool_name: str) -> bool:
        """Check if a tool is related to the file descriptor system.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool is part of the file descriptor system
        """
        return tool_name in self.fd_related_tools
    
    def register_fd_tool(self, tool_name: str) -> None:
        """Register a tool as being related to the file descriptor system.
        
        Args:
            tool_name: Name of the tool to register
        """
        self.fd_related_tools.add(tool_name)

    def create_fd(self, content: str, page_size: Optional[int] = None) -> Dict[str, Any]:
        """Create a new file descriptor for large content.

        Args:
            content: The content to store in the file descriptor
            page_size: Characters per page (defaults to default_page_size)

        Returns:
            Dictionary with file descriptor information
        """
        # Generate a sequential ID for the file descriptor
        fd_id = f"fd:{self.next_fd_id}"
        self.next_fd_id += 1  # Increment for next time
        
        # Use default page size if none provided
        page_size = page_size or self.default_page_size
        
        # Create line index for line-aware pagination
        lines, total_lines = self._index_lines(content)
        
        # We'll calculate the actual number of pages later with _calculate_total_pages
        # For now, use a rough estimate
        content_length = len(content)
        num_pages = (content_length + page_size - 1) // page_size  # Just a placeholder
        
        # Store the file descriptor entry with minimal info first
        self.file_descriptors[fd_id] = {
            "content": content,
            "lines": lines,  # Start indices of each line
            "total_lines": total_lines,
            "page_size": page_size,
            "creation_time": time.time(),
            "source": "tool_result",  # Default source, can be overridden
        }
        
        # Generate preview content (first page)
        preview_content, preview_info = self._get_page_content(fd_id, page=1)
        
        # Calculate the actual number of pages by simulating pagination
        num_pages = self._calculate_total_pages(fd_id)
        
        # Update the file descriptor with the calculated number of pages
        self.file_descriptors[fd_id]["total_pages"] = num_pages
        
        # Create the file descriptor result
        fd_result = {
            "fd": fd_id,
            "pages": num_pages,
            "truncated": preview_info.get("truncated", False),
            "lines": f"1-{preview_info.get('end_line', 1)}",
            "total_lines": total_lines,
            "message": f"Output exceeds {self.max_direct_output_chars} characters. Use read_fd to read more pages.",
            "preview": preview_content,
        }
        
        logger.debug(f"Created file descriptor {fd_id} with {num_pages} pages, {total_lines} lines")
        
        # Format the response in standardized XML format
        return self._format_fd_result(fd_result)

    def read_fd(
        self, 
        fd_id: str, 
        page: int = 1, 
        read_all: bool = False,
        extract_to_new_fd: bool = False
    ) -> Dict[str, Any]:
        """Read content from a file descriptor.

        Args:
            fd_id: The file descriptor ID to read from
            page: The page number to read (starting from 1)
            read_all: If True, returns the entire content
            extract_to_new_fd: If True, creates a new file descriptor with the content and returns its ID

        Returns:
            Dictionary with content and metadata, or a new file descriptor ID if extract_to_new_fd=True

        Raises:
            KeyError: If the file descriptor is not found
            ValueError: If the page number is invalid
        """
        # Validate file descriptor exists
        if fd_id not in self.file_descriptors:
            # Give a more helpful error for sequential FD numbering
            available_fds = ", ".join(sorted(self.file_descriptors.keys()))
            error_msg = f"File descriptor {fd_id} not found. Available FDs: {available_fds or 'none'}"
            logger.error(error_msg)
            return self._format_fd_error("not_found", fd_id, error_msg)
        
        fd_entry = self.file_descriptors[fd_id]
        
        # Prepare to get content based on read parameters
        content_to_return = None
        content_metadata = {}
        
        # Handle read_all case
        if read_all:
            # Use stored total pages value
            total_pages = fd_entry["total_pages"]
            
            content_to_return = fd_entry["content"]
            content_metadata = {
                "fd": fd_id,
                "page": "all",
                "pages": total_pages,
                "continued": False,
                "truncated": False,
                "lines": f"1-{fd_entry['total_lines']}",
                "total_lines": fd_entry["total_lines"],
            }
        else:
            # Use stored total pages value
            total_pages = fd_entry["total_pages"]
            
            # Validate page number
            if page < 1 or page > total_pages:
                error_msg = f"Invalid page number. Valid range: 1-{total_pages}"
                logger.error(error_msg)
                return self._format_fd_error("invalid_page", fd_id, error_msg)
            
            # Get content for the requested page
            content, page_info = self._get_page_content(fd_id, page)
            content_to_return = content
            
            # Create the response metadata
            content_metadata = {
                "fd": fd_id,
                "page": page,
                "pages": total_pages,
                "continued": page_info.get("continued", False),
                "truncated": page_info.get("truncated", False),
                "lines": f"{page_info.get('start_line', 1)}-{page_info.get('end_line', 1)}",
                "total_lines": fd_entry["total_lines"],
            }
            
            logger.debug(f"Read page {page}/{total_pages} from {fd_id}")
        
        # Check if we should extract the content to a new FD
        if extract_to_new_fd and content_to_return:
            # Create a new file descriptor with the content
            new_fd_result = self.create_fd(content_to_return)
            
            # Extract the new FD ID from the result
            new_fd_id = new_fd_result.content.split('fd="')[1].split('"')[0]
            
            # Return a special response indicating the content was extracted to a new FD
            return self._format_fd_extraction({
                "source_fd": fd_id,
                "new_fd": new_fd_id,
                "page": page if not read_all else "all",
                "content_size": len(content_to_return),
                "message": f"Content from {fd_id} has been extracted to {new_fd_id}",
            })
        
        # Add content to metadata and create the response
        content_metadata["content"] = content_to_return
        
        # Format the response in standardized XML format
        return self._format_fd_content(content_metadata)

    def write_fd_to_file(
        self, 
        fd_id: str, 
        file_path: str,
        mode: str = "write",
        create: bool = True,
        exist_ok: bool = True
    ) -> Dict[str, Any]:
        """Write file descriptor content to a file.
        
        Args:
            fd_id: The file descriptor ID
            file_path: Path to the file to write
            mode: "write" (default, overwrite) or "append" (add to existing file)
            create: Whether to create the file if it doesn't exist (default: True)
            exist_ok: Whether it's ok if the file already exists (default: True)
            
        Returns:
            ToolResult with success or error message
            
        Raises:
            FileNotFoundError: If the file descriptor doesn't exist
            PermissionError: If the file can't be written due to permissions
            IOError: If there's an I/O error writing the file
        """
        # Check if the file descriptor exists
        if fd_id not in self.file_descriptors:
            available_fds = ", ".join(sorted(self.file_descriptors.keys()))
            error_msg = f"File descriptor {fd_id} not found. Available FDs: {available_fds or 'none'}"
            logger.error(error_msg)
            return self._format_fd_error("not_found", fd_id, error_msg)
        
        # Get the content
        content = self.file_descriptors[fd_id]["content"]
        
        try:
            # Validate mode parameter
            if mode not in ["write", "append"]:
                error_msg = f"Invalid mode: {mode}. Valid options are 'write' or 'append'."
                logger.error(error_msg)
                return self._format_fd_error("invalid_parameter", fd_id, error_msg)
            
            # Check file existence
            file_path_obj = Path(file_path)
            file_exists = file_path_obj.exists()
            
            # Handle file existence according to parameters
            if file_exists and not exist_ok:
                error_msg = f"File {file_path} already exists and exist_ok=False"
                logger.error(error_msg)
                return self._format_fd_error("file_exists", fd_id, error_msg)
            
            if not file_exists and not create:
                error_msg = f"File {file_path} doesn't exist and create=False"
                logger.error(error_msg)
                return self._format_fd_error("file_not_found", fd_id, error_msg)
            
            # Ensure parent directory exists
            if not file_path_obj.parent.exists():
                file_path_obj.parent.mkdir(parents=True, exist_ok=True)
                
            # Open mode: 'w' for overwrite, 'a' for append
            file_mode = 'w' if mode == "write" else 'a'
            operation_type = "written" if mode == "write" else "appended"
            
            # Write the file
            with open(file_path, file_mode, encoding='utf-8') as f:
                f.write(content)
                
            # Create success message
            success_msg = f"File descriptor {fd_id} content ({len(content)} chars) successfully {operation_type} to {file_path}"
            logger.info(success_msg)
            
            # Format successful result
            result = {
                "fd": fd_id,
                "file_path": file_path,
                "mode": mode,
                "create": create,
                "exist_ok": exist_ok,
                "char_count": len(content),
                "size_bytes": os.path.getsize(file_path),
                "success": True,
                "message": success_msg
            }
            
            return self._format_fd_file_result(result)
            
        except Exception as e:
            # Handle any errors
            error_msg = f"Error writing file descriptor {fd_id} to {file_path}: {str(e)}"
            logger.error(error_msg)
            return self._format_fd_error("write_error", fd_id, error_msg)

    def _calculate_total_pages(self, fd_id: str) -> int:
        """Calculate the total number of pages in a file descriptor.
        
        This simulates the line-aware pagination algorithm to get an accurate page count.
        
        Args:
            fd_id: The file descriptor ID
            
        Returns:
            The total number of pages
        """
        if fd_id not in self.file_descriptors:
            return 0
            
        fd_entry = self.file_descriptors[fd_id]
        
        # If total_pages is already calculated, return it
        if "total_pages" in fd_entry:
            return fd_entry["total_pages"]
            
        # Always ensure total_pages is calculated and stored
        # This should happen during creation in most cases
            
        content = fd_entry["content"]
        page_size = fd_entry["page_size"]
        
        # For very small content, just return 1 page
        if len(content) <= page_size:
            return 1
            
        # For larger content, iterate through the pages
        start_char = 0
        page_count = 0
        
        while start_char < len(content):
            page_count += 1
            
            # Calculate end of current page
            end_char = min(start_char + page_size, len(content))
            
            # Find line boundaries for better pagination
            lines = fd_entry["lines"]
            end_line = 1
            
            # Find the end line for this page
            for i, line_start in enumerate(lines):
                if line_start >= end_char:
                    end_line = i  # The previous line
                    break
                end_line = i + 1
                
            # Determine the start of the next page
            if end_line < len(lines):
                start_char = lines[end_line]
            else:
                # No more lines, we're done
                break
        
        # Store the result in the fd_entry for future use
        fd_entry["total_pages"] = page_count
                
        return page_count

    def _index_lines(self, content: str) -> Tuple[List[int], int]:
        """Create an index of line start positions.

        Args:
            content: The content to index

        Returns:
            Tuple of (list of line start indices, total line count)
        """
        lines = [0]  # First line always starts at index 0
        for i, char in enumerate(content):
            if char == '\n' and i + 1 < len(content):
                lines.append(i + 1)
        
        return lines, len(lines)

    def _get_page_content(
        self, fd_id: str, page: int
    ) -> Tuple[str, Dict[str, Any]]:
        """Get content for a specific page with line-aware pagination.

        Args:
            fd_id: The file descriptor ID
            page: The page number (1-based)

        Returns:
            Tuple of (page content, page information)
        """
        fd_entry = self.file_descriptors[fd_id]
        content = fd_entry["content"]
        page_size = fd_entry["page_size"]
        lines = fd_entry["lines"]
        
        # Calculate page boundaries
        start_char = (page - 1) * page_size
        
        # Handle case where start_char is beyond the content length
        if start_char >= len(content):
            # Return empty content with info showing we're beyond content
            return "", {
                "start_line": fd_entry["total_lines"],
                "end_line": fd_entry["total_lines"],
                "continued": False,
                "truncated": False,
                "empty_page": True
            }
            
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
        
        # Check if we're continuing from previous page (not starting at line boundary)
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
        
        # Extract the actual content for this page
        page_content = content[start_char:end_char]
        
        # Return content and metadata
        page_info = {
            "start_line": start_line,
            "end_line": end_line,
            "continued": continued,
            "truncated": truncated,
        }
        
        return page_content, page_info

    def _format_fd_result(self, result: Dict[str, Any]) -> "ToolResult":
        """Format a file descriptor result in XML format.

        Args:
            result: Dictionary with file descriptor information

        Returns:
            ToolResult instance with formatted XML content
        """
        from llmproc.tools.tool_result import ToolResult
        
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

    def _format_fd_content(self, content: Dict[str, Any]) -> "ToolResult":
        """Format file descriptor content in XML format.

        Args:
            content: Dictionary with content and metadata

        Returns:
            ToolResult instance with formatted XML content
        """
        from llmproc.tools.tool_result import ToolResult
        
        # Handle the all-pages case differently
        if content["page"] == "all":
            xml = (
                f'<fd_content fd="{content["fd"]}" page="all" pages="{content["pages"]}" '
                f'continued="false" truncated="false" '
                f'lines="{content["lines"]}" total_lines="{content["total_lines"]}">\n'
                f'{content["content"]}\n'
                f'</fd_content>'
            )
        else:
            xml = (
                f'<fd_content fd="{content["fd"]}" page="{content["page"]}" '
                f'pages="{content["pages"]}" '
                f'continued="{str(content["continued"]).lower()}" '
                f'truncated="{str(content["truncated"]).lower()}" '
                f'lines="{content["lines"]}" total_lines="{content["total_lines"]}">\n'
                f'{content["content"]}\n'
                f'</fd_content>'
            )
        
        return ToolResult(content=xml, is_error=False)
    
    def _format_fd_file_result(self, result: Dict[str, Any]) -> "ToolResult":
        """Format file descriptor file operation result in XML format.
        
        Args:
            result: Dictionary with file operation result information
            
        Returns:
            ToolResult instance with formatted XML content
        """
        from llmproc.tools.tool_result import ToolResult
        
        # Include create and exist_ok attributes if present
        create_attr = f' create="{str(result.get("create", True)).lower()}"' if "create" in result else ""
        exist_ok_attr = f' exist_ok="{str(result.get("exist_ok", True)).lower()}"' if "exist_ok" in result else ""
        
        xml = (
            f'<fd_file_result fd="{result["fd"]}" file_path="{result["file_path"]}" '
            f'mode="{result["mode"]}" char_count="{result["char_count"]}" '
            f'size_bytes="{result["size_bytes"]}" success="{str(result["success"]).lower()}"'
            f'{create_attr}{exist_ok_attr}>\n'
            f'  <message>{result["message"]}</message>\n'
            f'</fd_file_result>'
        )
        
        return ToolResult(content=xml, is_error=False)

    def _format_fd_extraction(self, result: Dict[str, Any]) -> "ToolResult":
        """Format file descriptor extraction result in XML format.

        Args:
            result: Dictionary with extraction result information

        Returns:
            ToolResult instance with formatted XML content
        """
        from llmproc.tools.tool_result import ToolResult
        
        xml = (
            f'<fd_extraction source_fd="{result["source_fd"]}" new_fd="{result["new_fd"]}" '
            f'page="{result["page"]}" content_size="{result["content_size"]}">\n'
            f'  <message>{result["message"]}</message>\n'
            f'</fd_extraction>'
        )
        
        return ToolResult(content=xml, is_error=False)
        
    def _format_fd_error(
        self, error_type: str, fd_id: str, message: str
    ) -> "ToolResult":
        """Format a file descriptor error in XML format.

        Args:
            error_type: Type of error (e.g., "not_found", "invalid_page")
            fd_id: The file descriptor ID
            message: Error message

        Returns:
            ToolResult instance with formatted XML error content
        """
        from llmproc.tools.tool_result import ToolResult
        
        xml = (
            f'<fd_error type="{error_type}" fd="{fd_id}">\n'
            f'  <message>{message}</message>\n'
            f'</fd_error>'
        )
        
        return ToolResult(content=xml, is_error=True)


async def read_fd_tool(
    fd: str,
    page: int = 1,
    read_all: bool = False,
    extract_to_new_fd: bool = False,
    llm_process=None,
) -> "ToolResult":
    """Read content from a file descriptor.

    This system call allows an LLM to read content that was previously stored
    in a file descriptor due to its large size.

    Args:
        fd: The file descriptor ID to read from (e.g., "fd:12345")
        page: The page number to read (starting from 1)
        read_all: If true, returns the entire content
        extract_to_new_fd: If true, extracts the content to a new file descriptor
        llm_process: The LLMProcess instance with FD manager

    Returns:
        A ToolResult instance with the content and metadata,
        or a new file descriptor ID if extract_to_new_fd=True

    Raises:
        KeyError: If the file descriptor is not found
        ValueError: If the page number is invalid
    """
    from llmproc.tools.tool_result import ToolResult
    
    if not llm_process or not hasattr(llm_process, "fd_manager"):
        error_msg = "File descriptor operations require an LLMProcess with fd_manager"
        logger.error(f"READ_FD ERROR: {error_msg}")
        return ToolResult.from_error(error_msg)

    try:
        # Read from the file descriptor - should now return a ToolResult directly
        return llm_process.fd_manager.read_fd(
            fd, 
            page=page, 
            read_all=read_all,
            extract_to_new_fd=extract_to_new_fd
        )
    except Exception as e:
        error_msg = f"Error reading file descriptor: {str(e)}"
        logger.error(f"READ_FD ERROR: {error_msg}")
        logger.debug("Detailed traceback:", exc_info=True)
        return ToolResult.from_error(error_msg)


async def fd_to_file_tool(
    fd: str,
    file_path: str,
    mode: str = "write",
    create: bool = True,
    exist_ok: bool = True,
    llm_process=None,
) -> "ToolResult":
    """Write file descriptor content to a file.
    
    This system call exports the content of a file descriptor to a file on disk,
    making it accessible to other tools or processes.
    
    Args:
        fd: The file descriptor ID to export (e.g., "fd:12345")
        file_path: Path to the file to write
        mode: "write" (default, overwrite) or "append" (add to existing file)
        create: Whether to create the file if it doesn't exist (default: True)
        exist_ok: Whether it's ok if the file already exists (default: True)
        llm_process: The LLMProcess instance with FD manager
        
    Returns:
        A ToolResult instance with success or error information
        
    Raises:
        KeyError: If the file descriptor is not found
        PermissionError: If the file can't be written due to permissions
        IOError: If there's an I/O error writing the file
    """
    from llmproc.tools.tool_result import ToolResult
    
    if not llm_process or not hasattr(llm_process, "fd_manager"):
        error_msg = "File descriptor operations require an LLMProcess with fd_manager"
        logger.error(f"FD_TO_FILE ERROR: {error_msg}")
        return ToolResult.from_error(error_msg)
        
    try:
        # Write FD content to file - returns a ToolResult
        return llm_process.fd_manager.write_fd_to_file(
            fd, 
            file_path, 
            mode=mode,
            create=create,
            exist_ok=exist_ok
        )
    except Exception as e:
        error_msg = f"Error writing file descriptor to file: {str(e)}"
        logger.error(f"FD_TO_FILE ERROR: {error_msg}")
        logger.debug("Detailed traceback:", exc_info=True)
        return ToolResult.from_error(error_msg)