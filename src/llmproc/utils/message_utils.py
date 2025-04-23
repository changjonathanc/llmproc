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
    process.state.append({"role": role, "content": content, LLMPROC_MSG_ID: message_id})
    return message_id
