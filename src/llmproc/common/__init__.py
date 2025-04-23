"""Common shared components of llmproc.

This package contains fundamental data structures and utilities
that are used throughout the llmproc library. Components in this
package should have minimal dependencies to avoid circular imports.
"""

from llmproc.common.constants import LLMPROC_MSG_ID
from llmproc.common.results import ToolResult

__all__ = ["ToolResult", "LLMPROC_MSG_ID"]
