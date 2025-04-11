"""File descriptor system for managing large tool outputs.

This package implements a file descriptor (FD) system that handles the storage,
pagination, and access of large content that exceeds context limits. It provides
a Unix-like file descriptor abstraction for LLM processes.

Core components:
1. FileDescriptorManager - Central class managing file descriptors

Note: The actual tool implementations (read_fd, fd_to_file) are now in llmproc.tools.fd_tools
to avoid circular imports.
"""

from llmproc.file_descriptors.constants import (
    # Related tool names
    FD_RELATED_TOOLS,
    # System prompt sections
    FILE_DESCRIPTOR_INSTRUCTIONS,
    REFERENCE_INSTRUCTIONS,
    USER_INPUT_INSTRUCTIONS,
)
from llmproc.file_descriptors.manager import FileDescriptorManager

__all__ = [
    # Core components
    "FileDescriptorManager",
    # System prompt sections
    "FILE_DESCRIPTOR_INSTRUCTIONS",
    "USER_INPUT_INSTRUCTIONS",
    "REFERENCE_INSTRUCTIONS",
    # Related tool names
    "FD_RELATED_TOOLS",
]
