"""Callback system for LLMProc.

This module provides a minimal event-based callback system that allows
tracking and responding to events in LLMProcess execution.
"""

from enum import Enum


class CallbackEvent(Enum):
    """Enum of supported callback event types."""

    TOOL_START = "tool_start"  # Called when a tool execution starts
    TOOL_END = "tool_end"  # Called when a tool execution completes
    RESPONSE = "response"  # Called when model generates a response
    API_REQUEST = "api_request"  # Called when an API request is made
    API_RESPONSE = "api_response"  # Called when an API response is received
    TURN_START = "turn_start"  # Called at the start of each turn
    TURN_END = "turn_end"  # Called at the end of each turn
    STDERR_WRITE = "stderr_write"  # Called when text is appended to stderr log


__all__ = ["CallbackEvent"]
