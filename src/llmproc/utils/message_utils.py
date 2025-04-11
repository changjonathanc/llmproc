"""Utilities for message handling in LLMProcess."""


def append_message_with_id(process, role, content):
    """
    Append a message to the process state with an automatically generated GOTO ID.
    
    Args:
        process: The LLMProcess instance
        role: The message role (user/assistant)
        content: The message content
        
    Returns:
        The generated message ID
    """
    message_id = f"msg_{len(process.state)}"
    process.state.append({"role": role, "content": content, "goto_id": message_id})
    return message_id