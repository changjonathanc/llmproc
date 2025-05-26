"""Common constants used across multiple modules."""

# Message ID key used for message identification
# Used by GOTO tool, token caching, analytics, etc.
LLMPROC_MSG_ID = "llmproc_msg_id"

# Prefix used when rendering message IDs for the LLM, e.g. "msg_0".
# Keeping it in a central constants module avoids circular‑import risk
# between id_utils and tools that need to parse the formatted IDs.
MESSAGE_ID_PREFIX = "msg_"

# Tool metadata attribute name used to attach metadata to tool functions
# Double underscores plus the llmproc prefix virtually removes any
# collision risk with user‑defined attributes
TOOL_METADATA_ATTR = "__llmproc_tool_meta__"
