"""Core file descriptor management functionality.

This module contains the FileDescriptorManager class that centrally manages
file descriptors, their creation, access, and lifecycle within an LLMProcess.
"""

import copy
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

from llmproc.common.results import ToolResult
from llmproc.file_descriptors._docs_manager import (
    CREATE_FROM_TOOL_RESULT,
    EXTRACT_REFERENCES,
    FILEDESCRIPTORMANAGER_CLASS,
    HANDLE_USER_INPUT,
    INIT,
    IS_FD_RELATED_TOOL,
    PROCESS_REFERENCES,
    REGISTER_FD_TOOL,
)
from llmproc.file_descriptors.constants import FD_RELATED_TOOLS
from llmproc.file_descriptors.formatter import (
    format_fd_content,
    format_fd_error,
    format_fd_extraction,
    format_fd_file_result,
    format_fd_result,
)
from llmproc.file_descriptors.paginator import (
    calculate_total_pages,
    extract_content_by_mode,
    get_page_content,
    index_lines,
)
from llmproc.file_descriptors.references import (
    extract_references,
    format_user_input_reference,
)

# Set up logger
logger = logging.getLogger(__name__)


class FileDescriptorManager:
    """Manages file descriptors for large content."""

    def __init__(
        self,
        default_page_size: int = 4000,
        max_direct_output_chars: int = 8000,
        max_input_chars: int = 8000,
        page_user_input: bool = False,
        enable_references: bool = False,
    ):
        """Initialize the FileDescriptorManager."""
        self.file_descriptors: dict[str, dict[str, Any]] = {}
        self.default_page_size = default_page_size
        self.max_direct_output_chars = max_direct_output_chars
        self.max_input_chars = max_input_chars
        self.page_user_input = page_user_input
        self.enable_references = enable_references
        self.fd_related_tools = FD_RELATED_TOOLS.copy()
        self.next_fd_id = 1  # Counter for sequential FD IDs

    def is_fd_related_tool(self, tool_name: str) -> bool:
        """Check if a tool is related to the file descriptor system.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is part of the file descriptor system
        """
        return tool_name in self.fd_related_tools

    def create_fd_from_tool_result(self, content: str, tool_name: Optional[str] = None) -> tuple:
        """Create a file descriptor from tool result content if needed.

        Args:
            content: The tool result content
            tool_name: Optional tool name for skipping FD-related tools

        Returns:
            tuple: (fd_content, used_fd) where fd_content is either the original content
            or the FD result, and used_fd is a boolean indicating if FD was created
        """
        # Check if FD system should be used
        if (
            not isinstance(content, str)
            or (tool_name and self.is_fd_related_tool(tool_name))
            or len(content) <= self.max_direct_output_chars
        ):
            return content, False

        # Create FD for large content
        fd_xml = self.create_fd_content(content)
        # Wrap in ToolResult for consistent return type
        fd_result = ToolResult(content=fd_xml, is_error=False)
        return fd_result, True

    def register_fd_tool(self, tool_name: str) -> None:
        """Register a tool as being related to the file descriptor system.

        Args:
            tool_name: Name of the tool to register
        """
        self.fd_related_tools.add(tool_name)

    def create_fd_content(self, content: str, page_size: int | None = None, source: str = "tool_result") -> str:
        """Create a new file descriptor for large content.

        Args:
            content: The content to store in the file descriptor
            page_size: Characters per page (defaults to default_page_size)
            source: Source of the content (e.g., "tool_result", "user_input")

        Returns:
            Formatted XML string with file descriptor information
        """
        # Generate a sequential ID for the file descriptor
        fd_id = f"fd:{self.next_fd_id}"
        self.next_fd_id += 1  # Increment for next time

        # Use default page size if none provided
        page_size = page_size or self.default_page_size

        # Create line index for line-aware pagination
        lines, total_lines = index_lines(content)

        # Store the file descriptor entry with minimal info first
        self.file_descriptors[fd_id] = {
            "content": content,
            "lines": lines,  # Start indices of each line
            "total_lines": total_lines,
            "page_size": page_size,
            "creation_time": time.time(),
            "source": source,  # Source of the content
        }

        # Generate preview content (first page)
        preview_content, preview_info = get_page_content(content, lines, page_size, start_pos=1)

        # Calculate the actual number of pages by simulating pagination
        num_pages = calculate_total_pages(content, lines, page_size)

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
            "source": source,
        }

        logger.debug(f"Created file descriptor {fd_id} with {num_pages} pages, {total_lines} lines, source: {source}")

        # Format the response in standardized XML format
        return format_fd_result(fd_result)

    def read_fd_content(
        self,
        fd_id: str,
        read_all: bool = False,
        extract_to_new_fd: bool = False,
        mode: str = "page",
        start: int = 1,
        count: int = 1,
    ) -> str:
        """Read content from a file descriptor and return formatted XML string.

        Args:
            fd_id: The file descriptor ID to read from
            read_all: If True, returns the entire content
            extract_to_new_fd: If True, creates a new file descriptor with the content and returns its ID
            mode: Positioning mode: "page" (default), "line", or "char"
            start: Starting position in the specified mode's units (page number, line number, or character position)
            count: Number of units to read (pages, lines, or characters)

        Returns:
            Formatted XML string with content and metadata

        Raises:
            KeyError: If the file descriptor is not found
            ValueError: If the start position is invalid or if the range parameters are invalid
        """
        # Validate file descriptor exists
        if fd_id not in self.file_descriptors:
            # Give a more helpful error for sequential FD numbering
            available_fds = ", ".join(sorted(self.file_descriptors.keys()))
            error_msg = f"File descriptor {fd_id} not found. Available FDs: {available_fds or 'none'}"
            logger.error(error_msg)
            raise KeyError(error_msg)

        fd_entry = self.file_descriptors[fd_id]

        # Prepare to get content based on read parameters
        content_to_return = None
        content_metadata = {}

        # Validate mode parameter
        if mode not in ["page", "line", "char"]:
            error_msg = f"Invalid mode: {mode}. Valid options are 'page', 'line', or 'char'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Handle read_all case (highest priority)
        if read_all:
            # Read the entire content regardless of other positioning parameters
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
                "mode": "all",
            }

            logger.debug(f"Read all content from {fd_id}")

        # Handle positioning modes
        else:
            content_to_return, content_metadata = extract_content_by_mode(
                content=fd_entry["content"],
                lines=fd_entry["lines"],
                mode=mode,
                start=start,
                count=count,
                total_lines=fd_entry["total_lines"],
                page_size=fd_entry["page_size"],
                total_pages=fd_entry["total_pages"],
            )

            # Add fd_id to metadata
            content_metadata["fd"] = fd_id

        # Check if we should extract the content to a new FD
        if extract_to_new_fd and content_to_return:
            # Create a new file descriptor with the content
            new_fd_xml = self.create_fd_content(content_to_return)

            # Extract the new FD ID from the result
            # Note: This pattern is a bit fragile and depends on the XML format
            new_fd_id = new_fd_xml.split('fd="')[1].split('"')[0]

            # Return a special response indicating the content was extracted to a new FD
            extraction_result = {
                "source_fd": fd_id,
                "new_fd": new_fd_id,
                "mode": mode,
                "content_size": len(content_to_return),
                "message": f"Content from {fd_id} has been extracted to {new_fd_id}",
            }

            # Add mode-specific attributes based on the access mode
            if read_all:
                extraction_result["position"] = "all"
            else:
                extraction_result["start"] = start
                extraction_result["count"] = count

            return format_fd_extraction(extraction_result)

        # Add content to metadata and create the response
        content_metadata["content"] = content_to_return

        # Format the response in standardized XML format
        return format_fd_content(content_metadata)

    def write_fd_to_file_content(
        self,
        fd_id: str,
        file_path: str,
        mode: str = "write",
        create: bool = True,
        exist_ok: bool = True,
    ) -> str:
        """Write file descriptor content to a file and return formatted XML string.

        Args:
            fd_id: The file descriptor ID
            file_path: Path to the file to write
            mode: "write" (default, overwrite) or "append" (add to existing file)
            create: Whether to create the file if it doesn't exist (default: True)
            exist_ok: Whether it's ok if the file already exists (default: True)

        Returns:
            Formatted XML string with success information

        Raises:
            KeyError: If the file descriptor doesn't exist
            ValueError: If the parameters are invalid
            FileNotFoundError: If the file doesn't exist and create=False
            PermissionError: If the file can't be written due to permissions
            IOError: If there's an I/O error writing the file
        """
        # Check if the file descriptor exists
        if fd_id not in self.file_descriptors:
            available_fds = ", ".join(sorted(self.file_descriptors.keys()))
            error_msg = f"File descriptor {fd_id} not found. Available FDs: {available_fds or 'none'}"
            logger.error(error_msg)
            raise KeyError(error_msg)

        # Get the content
        content = self.file_descriptors[fd_id]["content"]

        # Validate mode parameter
        if mode not in ["write", "append"]:
            error_msg = f"Invalid mode: {mode}. Valid options are 'write' or 'append'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check file existence
        file_path_obj = Path(file_path)
        file_exists = file_path_obj.exists()

        # Handle file existence according to parameters
        if file_exists and not exist_ok:
            error_msg = f"File {file_path} already exists and exist_ok=False"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not file_exists and not create:
            error_msg = f"File {file_path} doesn't exist and create=False"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Ensure parent directory exists
        if not file_path_obj.parent.exists():
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Open mode: 'w' for overwrite, 'a' for append
        file_mode = "w" if mode == "write" else "a"
        operation_type = "written" if mode == "write" else "appended"

        # Write the file
        with open(file_path, file_mode, encoding="utf-8") as f:
            f.write(content)

        # Create success message
        success_msg = (
            f"File descriptor {fd_id} content ({len(content)} chars) successfully {operation_type} to {file_path}"
        )
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
            "message": success_msg,
        }

        return format_fd_file_result(result)

    def handle_user_input(self, user_input: str) -> str:
        """Handle large user input by creating a file descriptor if needed.

        Args:
            user_input: The user input to process

        Returns:
            The original user input if not paged, or a formatted FD reference
            if the input exceeds the threshold and paging is enabled
        """
        # Check if we should page this input
        if not self.page_user_input or len(user_input) <= self.max_input_chars:
            # No need to page this input
            return user_input

        # Create a file descriptor for the large user input
        fd_xml = self.create_fd_content(content=user_input, source="user_input")

        # Extract the FD ID from the result
        fd_id = fd_xml.split('fd="')[1].split('"')[0]

        # Format a user message that references the file descriptor
        formatted_message = format_user_input_reference(user_input, fd_id, max_preview_chars=self.max_input_chars // 20)

        return formatted_message

    def extract_references_from_message(self, assistant_message: str) -> list[dict[str, str]]:
        """Extract references from an assistant message and store in FD system."""
        if not self.enable_references:
            return []

        return extract_references(
            assistant_message=assistant_message,
            file_descriptors=self.file_descriptors,
            default_page_size=self.default_page_size,
            index_lines_func=index_lines,
        )

    def process_references(self, message: str) -> str:
        """Process references in an assistant message."""
        if not isinstance(message, str) or not self.enable_references:
            return message

        # Process references in the message
        references = self.extract_references_from_message(message)
        if references:
            logger.info(f"Extracted {len(references)} references from message")

        # The extraction process doesn't modify the message directly
        # It just stores the references in the FD system
        return message

    # ------------------------------------------------------------------
    # Cloning helper
    # ------------------------------------------------------------------

    def clone(self) -> "FileDescriptorManager":
        """Return a deep‑cloned copy of this manager for forked processes.

        This method creates a complete, independent copy of the file descriptor manager,
        including all file descriptors and settings. It's used specifically by the
        fork tool to ensure proper isolation between parent and child processes.

        Returns:
            A deep copy of this FileDescriptorManager with independent state
        """
        cloned = FileDescriptorManager(
            default_page_size=self.default_page_size,
            max_direct_output_chars=self.max_direct_output_chars,
            max_input_chars=self.max_input_chars,
            page_user_input=self.page_user_input,
            enable_references=self.enable_references,
        )

        cloned.file_descriptors = copy.deepcopy(self.file_descriptors)
        cloned.fd_related_tools = self.fd_related_tools.copy()
        cloned.next_fd_id = self.next_fd_id
        return cloned


# Apply full docstrings
FileDescriptorManager.__doc__ = FILEDESCRIPTORMANAGER_CLASS
FileDescriptorManager.__init__.__doc__ = INIT
FileDescriptorManager.handle_user_input.__doc__ = HANDLE_USER_INPUT
FileDescriptorManager.process_references.__doc__ = PROCESS_REFERENCES
FileDescriptorManager.extract_references_from_message.__doc__ = EXTRACT_REFERENCES
FileDescriptorManager.create_fd_from_tool_result.__doc__ = CREATE_FROM_TOOL_RESULT
FileDescriptorManager.register_fd_tool.__doc__ = REGISTER_FD_TOOL
FileDescriptorManager.is_fd_related_tool.__doc__ = IS_FD_RELATED_TOOL
