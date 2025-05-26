"""Utilities for message handling in LLMProcess."""

from llmproc.common.constants import LLMPROC_MSG_ID


def append_message_with_id(process, role, content):
    """
    Append a message to the process state with an automatically generated message ID.

    Args:
        process: The LLMProcess instance
        role: The message role (user/assistant)
        content: The message content

    Returns:
        The generated message ID (integer index)
    """
    message_id = len(process.state)  # Use integer index as message ID
    msg = {"role": role, "content": content}

    # Only add message ID if the user message and message IDs are enabled in the tool manager
    if (
        role == "user"
        and hasattr(process, "tool_manager")
        and getattr(process.tool_manager, "message_ids_enabled", False)
    ):
        msg[LLMPROC_MSG_ID] = message_id

    process.state.append(msg)
    return message_id
