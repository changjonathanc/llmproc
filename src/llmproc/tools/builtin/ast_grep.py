"""Wrapper tool for running ast-grep searches."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from llmproc.common.access_control import AccessLevel
from llmproc.common.results import ToolResult
from llmproc.tools.function_tools import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    description="Search source code using ast-grep patterns.",
    param_descriptions={
        "pattern": "The ast-grep search pattern to run.",
        "path": "File or directory to search. Defaults to the current directory.",
    },
    access=AccessLevel.READ,
)
async def ast_grep(pattern: str, path: str = ".") -> str | ToolResult:
    """Run ast-grep with the given pattern.

    Args:
        pattern: The pattern to search for.
        path: File or directory to search.

    Returns:
        Command output on success, otherwise a ``ToolResult`` with error details.
    """
    if not pattern:
        return ToolResult.from_error("Pattern must be provided.")

    command = shutil.which("ast-grep") or shutil.which("sg")
    if not command:
        logger.error("ast-grep executable not found")
        return ToolResult.from_error("ast-grep is not installed")

    search_path = str(Path(path))

    proc = await asyncio.create_subprocess_exec(
        command,
        pattern,
        search_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        error_msg = stderr.decode().strip() or f"ast-grep failed with code {proc.returncode}"
        logger.error("ast-grep error: %s", error_msg)
        return ToolResult.from_error(error_msg)

    return stdout.decode().strip()
