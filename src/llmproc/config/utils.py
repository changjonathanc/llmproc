"""Utility functions for program configuration."""

import os
from pathlib import Path


def resolve_path(
    file_path: str | Path,
    base_dir: Path | None = None,
    must_exist: bool = True,
    error_prefix: str = "File",
) -> Path:
    """Resolve a file path, optionally relative to a base directory.

    This is a common utility function used across the codebase for resolving
    file paths in a consistent way, especially for TOML configurations.

    Args:
        file_path: Path to resolve (can be relative or absolute)
        base_dir: Base directory for resolving relative paths
        must_exist: Whether to raise an error if the file doesn't exist
        error_prefix: Prefix for error messages (e.g., "System prompt file")

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If must_exist is True and the file doesn't exist
    """
    path = Path(file_path)

    # If path is not absolute and a base directory is provided, make it relative to base_dir
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path

    # Resolve to absolute path
    abs_path = path.resolve()

    # Check if the file exists if required
    if must_exist and not abs_path.exists():
        raise FileNotFoundError(
            f"{error_prefix} not found - Specified: '{file_path}', Resolved: '{os.path.abspath(file_path)}'"
        )

    return abs_path
