"""Tools for LLMProcess."""

from .spawn import spawn_tool
from .fork import fork_tool

__all__ = ["spawn_tool", "fork_tool"]