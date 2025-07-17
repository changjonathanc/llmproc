"""Unified plugin event definitions for hooks and callbacks."""

from __future__ import annotations

from enum import Enum


class EventCategory(Enum):
    """Classification for plugin events."""

    OBSERVATIONAL = "observational"
    BEHAVIORAL = "behavioral"


# ---------------------------------------------------------------------------
# Observational callback events
# ---------------------------------------------------------------------------
TOOL_START = "tool_start"
TOOL_END = "tool_end"
API_REQUEST = "api_request"
API_RESPONSE = "api_response"
API_STREAM_BLOCK = "api_stream_block"
TURN_START = "turn_start"
TURN_END = "turn_end"
RUN_END = "run_end"


# ---------------------------------------------------------------------------
# Behavioral hook method names
# ---------------------------------------------------------------------------
HOOK_USER_INPUT = "hook_user_input"
HOOK_TOOL_CALL = "hook_tool_call"
HOOK_TOOL_RESULT = "hook_tool_result"
HOOK_SYSTEM_PROMPT = "hook_system_prompt"
HOOK_RESPONSE = "hook_response"
HOOK_PROVIDE_TOOLS = "hook_provide_tools"


EVENT_CATEGORIES: dict[str | HookEvent, EventCategory] = {
    TOOL_START: EventCategory.OBSERVATIONAL,
    TOOL_END: EventCategory.OBSERVATIONAL,
    API_REQUEST: EventCategory.OBSERVATIONAL,
    API_RESPONSE: EventCategory.OBSERVATIONAL,
    API_STREAM_BLOCK: EventCategory.OBSERVATIONAL,
    TURN_START: EventCategory.OBSERVATIONAL,
    TURN_END: EventCategory.OBSERVATIONAL,
    RUN_END: EventCategory.OBSERVATIONAL,
    HOOK_USER_INPUT: EventCategory.BEHAVIORAL,
    HOOK_TOOL_CALL: EventCategory.BEHAVIORAL,
    HOOK_TOOL_RESULT: EventCategory.BEHAVIORAL,
    HOOK_SYSTEM_PROMPT: EventCategory.BEHAVIORAL,
    HOOK_RESPONSE: EventCategory.BEHAVIORAL,
    HOOK_PROVIDE_TOOLS: EventCategory.BEHAVIORAL,
}


class CallbackEvent(Enum):
    """Enum of supported callback events."""

    TOOL_START = TOOL_START
    TOOL_END = TOOL_END
    API_REQUEST = API_REQUEST
    API_RESPONSE = API_RESPONSE
    API_STREAM_BLOCK = API_STREAM_BLOCK
    TURN_START = TURN_START
    TURN_END = TURN_END
    RUN_END = RUN_END


class HookEvent(Enum):
    """Enum of supported hook events."""

    USER_INPUT = HOOK_USER_INPUT
    TOOL_CALL = HOOK_TOOL_CALL
    TOOL_RESULT = HOOK_TOOL_RESULT
    SYSTEM_PROMPT = HOOK_SYSTEM_PROMPT
    RESPONSE = HOOK_RESPONSE
    PROVIDE_TOOLS = HOOK_PROVIDE_TOOLS


# Support querying by HookEvent members as well as string constants
EVENT_CATEGORIES.update(
    {
        HookEvent.USER_INPUT: EventCategory.BEHAVIORAL,
        HookEvent.TOOL_CALL: EventCategory.BEHAVIORAL,
        HookEvent.TOOL_RESULT: EventCategory.BEHAVIORAL,
        HookEvent.SYSTEM_PROMPT: EventCategory.BEHAVIORAL,
        HookEvent.RESPONSE: EventCategory.BEHAVIORAL,
        HookEvent.PROVIDE_TOOLS: EventCategory.BEHAVIORAL,
    }
)


__all__ = [
    "EventCategory",
    "EVENT_CATEGORIES",
    "CallbackEvent",
    "HookEvent",
    "TOOL_START",
    "TOOL_END",
    "API_REQUEST",
    "API_RESPONSE",
    "API_STREAM_BLOCK",
    "TURN_START",
    "TURN_END",
    "RUN_END",
    "HOOK_USER_INPUT",
    "HOOK_TOOL_CALL",
    "HOOK_TOOL_RESULT",
    "HOOK_SYSTEM_PROMPT",
    "HOOK_RESPONSE",
    "HOOK_PROVIDE_TOOLS",
]
