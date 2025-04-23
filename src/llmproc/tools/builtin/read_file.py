"""Simple read_file tool for demonstration purposes."""

import logging
import os
from pathlib import Path
from typing import Any

from llmproc.common.access_control import AccessLevel
from llmproc.common.results import ToolResult
from llmproc.tools.function_tools import register_tool

# Set up logger
logger = logging.getLogger(__name__)


@register_tool(
    description="Reads a file from the file system and returns its contents.",
    param_descriptions={
        "file_path": "Absolute or relative path to the file to read. For security reasons, certain directories may be inaccessible."
    },
    access=AccessLevel.READ,
)
async def read_file(file_path: str) -> str:
    """Read a file and return its contents.

    Args:
        file_path: Path to the file to read

    Returns:
        The file contents as a string
    """
    try:
        # Normalize the path
        path = Path(file_path)
        if not os.path.isabs(file_path):
            # Make relative paths relative to current working directory
            path = Path(os.getcwd()) / path

        # Check if the file exists
        if not path.exists():
            error_msg = f"File not found: {path}"
            logger.error(error_msg)
            return ToolResult.from_error(error_msg)

        # Read the file
        content = path.read_text()

        # Return the content
        return content
    except Exception as e:
        error_msg = f"Error reading file {file_path}: {str(e)}"
        logger.error(error_msg)
        return ToolResult.from_error(error_msg)
