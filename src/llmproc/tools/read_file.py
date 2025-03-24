"""Simple read_file tool for demonstration purposes."""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from llmproc.tools.tool_result import ToolResult

# Set up logger
logger = logging.getLogger(__name__)

# Tool descriptions
read_file_tool_description = """
Reads a file from the file system and returns its contents.

Usage:
  read_file(file_path="/path/to/file.txt")

Parameters:
  file_path (str): The path to the file to read

When to use this tool:
- When you need to view the content of a file
- When you want to analyze text content from the file system
- For demonstration of file descriptor system with large files
"""

# Tool definitions for Anthropic API
read_file_tool_def = {
    "name": "read_file",
    "description": read_file_tool_description,
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path to the file to read",
            },
        },
        "required": ["file_path"],
    },
}


async def read_file_tool(file_path: str) -> Dict[str, Any]:
    """Read a file and return its contents.

    This is a simple tool for demonstration purposes. It reads a file from the
    file system and returns its contents. It's especially useful for demonstrating
    the file descriptor system with large files.

    Args:
        file_path: Path to the file to read

    Returns:
        A dictionary with the file contents or an error message
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
        return ToolResult.from_success(content)
    except Exception as e:
        error_msg = f"Error reading file {file_path}: {str(e)}"
        logger.error(error_msg)
        return ToolResult.from_error(error_msg)