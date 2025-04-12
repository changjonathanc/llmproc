"""GOTO Tool for LLMProcess.

This module provides a time travel tool that allows LLMs to reset the conversation
to a previous point, enabling self-correction without user intervention.
"""

import datetime
import logging
from typing import Any, Optional

from llmproc.common.results import ToolResult
from llmproc.utils.message_utils import append_message_with_id
from llmproc.tools.context_aware import context_aware

# Set up logger
logger = logging.getLogger(__name__)

# Tool definition that will be registered with the tool registry
GOTO_TOOL_DEFINITION = {
    "name": "goto",
    "description": """Reset the conversation to a previous point using a message ID. This tool enables "time travel" capabilities, allowing you to discard the most recent messages and start over from a previous point in time.

WHEN TO USE THIS TOOL:
- You finished a task that consumed a lot of context window but you don't need to keep the intermediate steps.
- You tried something but it didn't work, and you want to go back and try a different approach while compacting the conversation history.
- The user explicitly asks you to go back in the conversation

HOW IT WORKS:
- Each message will have a unique ID shown as [msg_X] at the start of the message
- Specify which message to return to using its ID (e.g., "msg_0" for the very beginning)
- Provide a new message that will replace all messages after that point. This is your chance to summarize what you've done and what you'll do next.


WHAT HAPPENS AFTER USING GOTO:
- The system will RESET the conversation history to the specified point
- All messages after that point will be deleted from history
- Your "message" parameter will be split into system note and assistant message parts using XML tags
- In your next turn, you should completely disregard any previous context and focus only on the new topic
- The conversation history has been reset, so never reference topics from before the GOTO


Before using GOTO:
[msg_0..6] some conversation
[msg_7] User: can you fix the depreciation warnings in the tests?
[msg_8..20] Assistant: (executes tool to fix depreciation warnings)
[msg_21] User: let's compact the history and focus on the other issue next.
[msg_22] Assistant: calls goto(position="msg_7", message="we've successfully fixed the depreciation warnings. and user wants to focus on the other issue next.")


EXAMPLE OF WHAT YOU'LL SEE:
If you see this, it means the GOTO tool was used. You should infer from the system message and assistant message and continue.

[msg_0..6] some conversation
[msg_7] User:
<system_message>
GOTO tool used. Conversation reset to message msg_7. 17 messages were removed.
</system_message>
<original_message_to_be_ignored>
can you fix the depreciation warnings in the tests?
</original_message_to_be_ignored>
<time_travel_message>
we've successfully fixed the depreciation warnings. and user wants to focus on the other issue next.
</time_travel_message>

you can either
1. simply acknowledging the GOTO and wait for new user input
2. if there's explicit instruction in the time travel message, follow it


NOTE: This tool performs a COMPLETE RESET of the conversation context to the specified point.
It is like starting a new conversation from that point. All context from messages after the reset point
is completely removed and should be considered forgotten.
""",
    "input_schema": {
        "type": "object",
        "properties": {
            "position": {
                "type": "string",
                "description": "Message ID to go back to (e.g., msg_3)"
            },
            "message": {
                "type": "string",
                "description": "Detailed message explaining why you're going back and what new approach you'll take (or summarizing what was accomplished)."
            }
        },
        "required": ["position", "message"]
    }
}


def find_position_by_id(state, message_id):
    """Find a message position in conversation history by its ID.

    Args:
        state: Conversation state
        message_id: The message ID to find (e.g., "msg_3")

    Returns:
        Index of the found message or None
    """
    # Handle empty state
    if not state:
        return None

    # Validate message_id format
    if not isinstance(message_id, str) or not message_id.startswith("msg_"):
        return None

    # Special case for "msg_0" - always return first message if it exists
    if message_id == "msg_0" and state:
        return 0

    # First try direct lookup using the goto_id field in message metadata
    for i, msg in enumerate(state):
        if msg.get("goto_id") == message_id:
            return i

    # If not found, try numeric extraction as a fallback
    try:
        msg_num = int(message_id.split('_')[1])
        if 0 <= msg_num < len(state):
            return msg_num
    except (IndexError, ValueError):
        pass

    # Message ID not found
    return None


# Using append_message_with_id from llmproc.utils.message_utils


@context_aware
async def handle_goto(
    position: str,
    message: str,
    runtime_context: Optional[Any] = None
):
    """Reset conversation to a previous point identified by message ID.

    Args:
        position: Message ID to go back to (e.g., msg_3)
        message: Detailed message explaining why you're going back and what new approach you'll take
        runtime_context: Runtime context dictionary containing dependencies needed by the tool.
            Required keys: 'process' (LLMProcess instance)

    Returns:
        ToolResult with success or error information
    """
    # Get process from runtime context
    if not runtime_context or "process" not in runtime_context:
        return ToolResult.from_error("Missing process in runtime_context - cannot perform time travel")

    process = runtime_context["process"]

    # Define error message templates
    error_messages = {
        "invalid_id_format": "Invalid message ID: {}. Must be in format 'msg_X' where X is a message number.",
        "id_not_found": "Could not find message with ID: {}. Available IDs range from msg_0 to msg_{}.",
        "cannot_go_forward": "Cannot go forward in time. Message {} is at or beyond the current point."
    }

    if not position or not position.startswith("msg_"):
        return ToolResult.from_error(error_messages["invalid_id_format"].format(position))

    # Find target position in history by message ID
    target_index = find_position_by_id(process.state, position)
    if target_index is None:
        # Show available range in error message
        max_id = len(process.state) - 1
        return ToolResult.from_error(error_messages["id_not_found"].format(position, max_id))

    # Check if trying to go forward instead of backward
    if target_index >= len(process.state) - 1:
        return ToolResult.from_error(error_messages["cannot_go_forward"].format(position))

    # Log the operation
    logger.info(f"GOTO: Resetting conversation from {len(process.state)} messages to {target_index+1} messages")

    # Store time travel metadata in process
    if not hasattr(process, "time_travel_history"):
        process.time_travel_history = []

    process.time_travel_history.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "from_message_count": len(process.state),
        "to_message_count": target_index + 1,
        "position_reference": position
    })

    # Debug the truncation
    logger.info(f"Before truncation, state has {len(process.state)} messages")
    logger.info(f"Will keep messages up to index {target_index}")

    # Truncate history after target
    original_content = process.state[target_index]["content"]
    process.state = process.state[:target_index]

    logger.info(f"After truncation, state has {len(process.state)} messages")

    # Optionally add new message
    if message:
        # Use the message directly
        # Calculate the number of messages that were removed (this was before truncation)
        original_message_count = process.time_travel_history[-1]["from_message_count"]
        removed_message_count = original_message_count - (target_index + 1)

        # Format the system note with system_message tags
        system_note = f"<system_message> GOTO tool used. Conversation reset to message {position}. {removed_message_count} messages were removed. </system_message>"
        original_message = f"<original_message_to_be_ignored>\n{original_content}\n</original_message_to_be_ignored>"
        # Format the assistant message with time_travel_message tags
        formatted_message = f"<time_travel_message>\n{message}\n</time_travel_message>"

        # Combine the system note with the user's message
        final_message = f"{system_note}\n{original_message}\n{formatted_message}"

        # Use append_message_with_id to ensure it gets a proper ID
        append_message_with_id(process, "user", final_message)

        return ToolResult.from_success(
            f"Conversation reset to message {position}. Added time travel message."
        )
    else:
        return ToolResult.from_success(
            f"Conversation reset to message {position}."
        )