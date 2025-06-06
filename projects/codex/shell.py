"""Minimal Docker Shell Session implementation following SPEC.md"""

# Import all shell types from the new package structure
from shells import APIShell, BashNotFoundError, CodexShell, InteractiveShell

# Backward compatibility
Shell = APIShell

__all__ = ["APIShell", "InteractiveShell", "CodexShell", "Shell", "BashNotFoundError"]
