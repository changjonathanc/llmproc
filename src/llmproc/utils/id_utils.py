"""Utilities for message ID formatting.

This module provides utilities for consistent formatting of message IDs
across the codebase, making it easy to change the display format in one place.
"""

# Import prefix constant from central location to avoid circular deps
from llmproc.common.constants import MESSAGE_ID_PREFIX

# Configuration settings
# The only public configuration knob – controls ID rendering style.
DEFAULT_ID_STYLE = "brackets"  # Options: "brackets" | "xml"


def render_id(msg_id) -> str:
    """Return the human/LLM-visible representation of a message id.

    Args:
        msg_id: The message ID (integer index or string)

    Returns:
        Formatted message ID string ready for display

    Raises:
        ValueError: If DEFAULT_ID_STYLE is set to an unknown value
    """
    # Convert integer IDs to the standard format with prefix
    if isinstance(msg_id, int):
        id_str = f"{MESSAGE_ID_PREFIX}{msg_id}"
    else:
        # For any string IDs, use as-is
        id_str = str(msg_id)

    # Format based on style setting
    if DEFAULT_ID_STYLE == "brackets":
        return f"[{id_str}] "
    if DEFAULT_ID_STYLE == "xml":
        return f"<message_id>{id_str}</message_id> "
    raise ValueError(f"Unknown id style: {DEFAULT_ID_STYLE}")
