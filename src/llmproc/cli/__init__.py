"""Command-line interface for LLMProc.

This module provides the CLI functionality for the llmproc package.
The main entry point is the 'main' function from the demo module.
"""

# Import the main entry point - but import it lazily to avoid circular imports
from llmproc.cli.demo import main

# Export symbols used by downstream code
__all__ = ["main"]
