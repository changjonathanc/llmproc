"""Tool for writing messages to process stderr.

This tool appends a message to a list buffer provided via the runtime
context under the ``"stderr"`` key. Each call simply appends the text
to the list to mimic how an LLM emits incremental log output.
"""

from typing import Any, Optional

from llmproc.callbacks import CallbackEvent
from llmproc.common.access_control import AccessLevel
from llmproc.common.results import ToolResult
from llmproc.tools.function_tools import register_tool


@register_tool(
    name="write_stderr",
    description="Append a message to the process stderr buffer.",
    param_descriptions={},
    requires_context=True,
    required_context_keys=["process", "stderr"],
    access=AccessLevel.READ,  # does not interact with external environment
)
async def write_stderr_tool(message: str, runtime_context: Optional[dict[str, Any]] = None) -> ToolResult:
    """Append a message to the stderr buffer.

    Args:
        message: Text to append
        runtime_context: Context containing the ``stderr`` buffer

    Returns:
        ToolResult indicating success or failure
    """
    if not message or not isinstance(message, str):
        return ToolResult.from_error("message must be a non-empty string")

    stderr: list[str] = runtime_context["stderr"]
    process = runtime_context["process"]

    stderr.append(message)
    process.trigger_event(CallbackEvent.STDERR_WRITE, message)
    return ToolResult.from_success("written")
