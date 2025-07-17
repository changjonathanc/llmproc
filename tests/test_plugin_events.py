"""Tests for unified plugin event definitions."""

from llmproc.plugin.events import (
    API_REQUEST,
    API_RESPONSE,
    API_STREAM_BLOCK,
    EVENT_CATEGORIES,
    HOOK_PROVIDE_TOOLS,
    HOOK_RESPONSE,
    HOOK_SYSTEM_PROMPT,
    HOOK_TOOL_CALL,
    HOOK_TOOL_RESULT,
    HOOK_USER_INPUT,
    RUN_END,
    TOOL_END,
    TOOL_START,
    TURN_END,
    TURN_START,
    CallbackEvent,
    EventCategory,
    HookEvent,
)


def test_callback_event_values_match_strings():
    """Enum values should match the string constants."""
    assert CallbackEvent.TOOL_START.value == TOOL_START
    assert CallbackEvent.RUN_END.value == RUN_END
    assert CallbackEvent.API_STREAM_BLOCK.value == API_STREAM_BLOCK


def test_hook_event_enum_values_match_strings():
    """HookEvent enum values should match their string constants."""
    assert HookEvent.USER_INPUT.value == HOOK_USER_INPUT
    assert HookEvent.TOOL_CALL.value == HOOK_TOOL_CALL
    assert HookEvent.TOOL_RESULT.value == HOOK_TOOL_RESULT
    assert HookEvent.SYSTEM_PROMPT.value == HOOK_SYSTEM_PROMPT
    assert HookEvent.RESPONSE.value == HOOK_RESPONSE
    assert HookEvent.PROVIDE_TOOLS.value == HOOK_PROVIDE_TOOLS


def test_event_classification():
    """plugin_events categorizes each event correctly."""
    assert EVENT_CATEGORIES[TOOL_START] is EventCategory.OBSERVATIONAL
    assert EVENT_CATEGORIES[API_STREAM_BLOCK] is EventCategory.OBSERVATIONAL
    assert EVENT_CATEGORIES[TOOL_END] is EventCategory.OBSERVATIONAL
    assert EVENT_CATEGORIES[RUN_END] is EventCategory.OBSERVATIONAL
    assert EVENT_CATEGORIES[TURN_START] is EventCategory.OBSERVATIONAL
    assert EVENT_CATEGORIES[TURN_END] is EventCategory.OBSERVATIONAL
    assert EVENT_CATEGORIES[API_REQUEST] is EventCategory.OBSERVATIONAL
    assert EVENT_CATEGORIES[API_RESPONSE] is EventCategory.OBSERVATIONAL
    assert EVENT_CATEGORIES[HookEvent.USER_INPUT] is EventCategory.BEHAVIORAL
    assert EVENT_CATEGORIES[HookEvent.TOOL_CALL] is EventCategory.BEHAVIORAL
    assert EVENT_CATEGORIES[HookEvent.TOOL_RESULT] is EventCategory.BEHAVIORAL
    assert EVENT_CATEGORIES[HookEvent.SYSTEM_PROMPT] is EventCategory.BEHAVIORAL
    assert EVENT_CATEGORIES[HookEvent.RESPONSE] is EventCategory.BEHAVIORAL
    assert EVENT_CATEGORIES[HookEvent.PROVIDE_TOOLS] is EventCategory.BEHAVIORAL
