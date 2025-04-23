"""Callback system for LLMProc.

This module provides a minimal event-based callback system that allows
tracking and responding to events in LLMProcess execution.
"""

from enum import Enum

class CallbackEvent(Enum):
    """Enum of supported callback event types."""
    TOOL_START = "tool_start"  # Called when a tool execution starts
    TOOL_END = "tool_end"      # Called when a tool execution completes
    RESPONSE = "response"      # Called when model generates a response

__all__ = ["CallbackEvent"]
