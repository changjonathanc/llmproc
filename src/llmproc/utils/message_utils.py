"""Utilities for message handling in LLMProcess."""


def append_message(process, role, content):
    """Append a message to the process state.

    The ``MessageIDPlugin`` handles message ID prefixing via user input hooks.

    Args:
        process: The ``LLMProcess`` instance.
        role: The message role (``user`` or ``assistant``).
        content: The message content.
    """
    msg = {"role": role, "content": content}

    process.state.append(msg)
