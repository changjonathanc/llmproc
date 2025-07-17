"""Docstrings for the FileDescriptorManager class and its methods."""

# Class docstring
FILEDESCRIPTORMANAGER_CLASS = """Manages file descriptors for large content.

This class maintains a registry of active file descriptors, handling
creation, reading, and pagination of content that exceeds context limits.
It provides a centralized interface for all file descriptor operations,
including content pagination, reference extraction, and file operations.

Attributes:
    file_descriptors (dict): Dictionary mapping fd IDs to descriptor entries
    default_page_size (int): Default character count per page
    max_direct_output_chars (int): Threshold for automatic FD creation
    max_input_chars (int): Threshold for automatic user input FD creation
    page_user_input (bool): Whether to automatically page large user inputs
    enable_references (bool): Whether to enable reference ID system
    fd_related_tools (set): Set of tool names that are part of the FD system
"""

# Method docstrings
INIT = """Initialize the FileDescriptorManager.

Args:
    default_page_size: Default number of characters per page
    max_direct_output_chars: Threshold for automatic FD creation
    max_input_chars: Threshold for automatic user input FD creation
    page_user_input: Whether to automatically page large user inputs
    enable_references: Whether to enable the reference ID system
"""


HANDLE_USER_INPUT = """Handle large user input by creating a file descriptor if needed.

Args:
    user_input: The user input to process

Returns:
    The original user input if not paged, or a formatted FD reference
    if the input exceeds the threshold and paging is enabled
"""

PROCESS_REFERENCES = """Process references in an assistant message.

Detects and processes reference patterns in assistant messages,
storing referenced sections as file descriptors for efficient retrieval.

Args:
    message: The message to process for references

Returns:
    The processed message with reference IDs properly formatted
"""

EXTRACT_REFERENCES = """Extract references from an assistant message and store in FD system.

Args:
    assistant_message: The assistant's message to process

Returns:
    List of reference information dictionaries
"""

CREATE_FROM_TOOL_RESULT = """Create a file descriptor from tool result content if needed.

Args:
    content: The tool result content
    tool_name: Optional tool name for skipping FD-related tools

Returns:
    tuple: (fd_content, used_fd) where fd_content is either the original content
    or the FD result, and used_fd is a boolean indicating if FD was created
"""

REGISTER_FD_TOOL = """Register a tool as being related to the file descriptor system.

Args:
    tool_name: Name of the tool to register
"""

IS_FD_RELATED_TOOL = """Check if a tool is related to the file descriptor system.

Args:
    tool_name: Name of the tool to check

Returns:
    True if the tool is part of the file descriptor system
"""
