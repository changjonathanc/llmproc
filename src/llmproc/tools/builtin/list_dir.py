"""Directory listing tool for LLMProcess."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from llmproc.common.access_control import AccessLevel
from llmproc.common.results import ToolResult
from llmproc.tools.function_tools import register_tool

# Set up logger
logger = logging.getLogger(__name__)


@register_tool(
    description="Lists directory contents with options for showing hidden files and detailed information.",
    param_descriptions={
        "directory_path": "Absolute or relative path to the directory to list. Defaults to current working directory if not specified.",
        "show_hidden": "Whether to include hidden files and directories in the listing. Defaults to False.",
        "detailed": "Whether to show detailed information (size, permissions, modification time) for each item. Defaults to False.",
    },
    access=AccessLevel.READ,
)
async def list_dir(directory_path: str = ".", show_hidden: bool = False, detailed: bool = False) -> str:
    """List directory contents with options for showing hidden files and detailed information.

    Args:
        directory_path: Path to the directory to list. Defaults to current directory.
        show_hidden: Whether to include hidden files and directories. Defaults to False.
        detailed: Whether to show detailed information for each item. Defaults to False.

    Returns:
        A formatted string of directory contents
    """
    try:
        # Normalize the path
        path = Path(directory_path)
        if not os.path.isabs(directory_path):
            # Make relative paths relative to current working directory
            path = Path(os.getcwd()) / path

        # Check if the directory exists
        if not path.exists():
            error_msg = f"Directory not found: {path}"
            logger.error(error_msg)
            return ToolResult.from_error(error_msg)

        # Check if it's a directory
        if not path.is_dir():
            error_msg = f"Path is not a directory: {path}"
            logger.error(error_msg)
            return ToolResult.from_error(error_msg)

        # Get directory contents
        dir_items = []
        for item in path.iterdir():
            # Skip hidden files if show_hidden is False
            if not show_hidden and item.name.startswith("."):
                continue

            if detailed:
                # Get file stats
                stats = item.stat()
                # Format size to be more readable
                size = _format_size(stats.st_size)
                # Format last modified time
                mtime = _format_time(stats.st_mtime)
                # Get type indicator (directory, file, etc.)
                type_indicator = "d" if item.is_dir() else "f"
                # Add detailed item info
                dir_items.append(f"{type_indicator} {item.name} (Size: {size}, Modified: {mtime})")
            else:
                # Add simple item name with type indicator
                dir_items.append(f"{'d' if item.is_dir() else 'f'} {item.name}")

        # Sort items (directories first, then files)
        dir_items.sort()

        # Format the output
        if not dir_items:
            return f"Directory '{path}' is empty."

        result = f"Contents of '{path}':\n" + "\n".join(dir_items)
        return result

    except Exception as e:
        error_msg = f"Error listing directory {directory_path}: {str(e)}"
        logger.error(error_msg)
        return ToolResult.from_error(error_msg)


def _format_size(size_bytes: int) -> str:
    """Format file size in a human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def _format_time(timestamp: float) -> str:
    """Format timestamp as a human-readable date/time."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")
