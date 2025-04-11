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
    "description": """Reset the conversation to a previous point using a message ID. This tool enables "time travel" capabilities, allowing recovery from conversational dead-ends or misunderstandings.

WHEN TO USE THIS TOOL:
- The user explicitly asks you to go back in the conversation
- The conversation has gone off-track or down an unproductive path
- You want to restart from an earlier point with a different strategy
- You want to compress context after multiple tool operations

IMPORTANT: Only use this tool ONCE per response. Never call GOTO multiple times in succession.

HOW IT WORKS:
- Each message has a unique ID shown as [msg_X] at the start of the message
- Specify which message to return to using its ID (e.g., "msg_0" for the very beginning)
- Provide a new message that will replace all messages after that point

CRITICAL INSTRUCTIONS:
1. When asked to "go back to the beginning", always use "msg_0" as the position.
2. Include a clear, concise message explaining why you're using GOTO and what you'll do next.
3. After using GOTO, your next response should directly address the user's request.
4. Never use GOTO multiple times in succession - it must be used at most once per turn.

WHAT HAPPENS AFTER USING GOTO:
- The system will RESET the conversation history to the specified point
- All messages after that point will be deleted from history
- Your "message" parameter text will be added with special <time_travel> XML tags
- In your next turn, you'll see the truncated history with your special message
- You should respond directly to the user's request WITHOUT referring to the time travel

EXAMPLE OF WHAT YOU'LL SEE:

Before using goto:
[msg_0] User: Hello, what can you help me with?
[msg_1] Assistant: I can help with reading files, calculations, and more.
[msg_2] User: Can you explain how black holes work?
[msg_3] Assistant: Black holes are regions of spacetime where gravity is so strong...

After you call: goto(position="msg_0", message="Let's start over and talk about AI instead.")

You'll see something like this:
[msg_0] User: Hello, what can you help me with?
[msg_1] User: [SYSTEM NOTE: Conversation reset to message msg_0. 3 messages were removed.]

<time_travel>
Let's start over and talk about AI instead.
</time_travel>

Then you should respond directly about AI without mentioning the time travel.

NOTE: After using this tool, the conversation will continue from the specified point,
and messages after that point will be forgotten. Only use this tool when necessary.
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
    process.state = process.state[:target_index+1]
    
    logger.info(f"After truncation, state has {len(process.state)} messages")

    # Optionally add new message
    if message:
        # Always wrap the message in time_travel tags (removing existing ones if present)
        clean_message = message

        # Remove existing tags if present to avoid duplication
        if "<time_travel>" in clean_message and "</time_travel>" in clean_message:
            clean_message = clean_message.replace("<time_travel>", "").replace("</time_travel>", "")

        # Wrap with time_travel tags
        formatted_message = f"<time_travel>\n{clean_message.strip()}\n</time_travel>"

        # Add a system note about the time travel
        system_note = f"[SYSTEM NOTE: Conversation reset to message {position}. {len(process.state) - (target_index + 1)} messages were removed.]"

        # Combine the system note with the user's message
        final_message = f"{system_note}\n\n{formatted_message}"

        # Use append_message_with_id to ensure it gets a proper ID
        append_message_with_id(process, "user", final_message)

        return ToolResult.from_success(
            f"Conversation reset to message {position}. Added time travel message."
        )
    else:
        return ToolResult.from_success(
            f"Conversation reset to message {position}."
        )
