from __future__ import annotations

import logging
from typing import Any, Optional

from llmproc.common.results import ToolResult
from llmproc.config.schema import StderrPluginConfig
from llmproc.plugins.override_utils import apply_tool_overrides
from llmproc.tools.function_tools import register_tool

logger = logging.getLogger(__name__)


class StderrPlugin:
    """Plugin that captures messages written to process stderr."""

    def __init__(self, config: StderrPluginConfig | None = None) -> None:
        self.config = config or StderrPluginConfig()
        self.log: list[str] = []

    def fork(self) -> StderrPlugin:
        """Return ``self`` for a forked process."""
        return self

    def get_log(self) -> list[str]:
        """Return a copy of the accumulated stderr log."""
        return self.log.copy()

    def clear_log(self) -> None:
        """Clear the stderr log."""
        self.log.clear()

    @register_tool(
        name="write_stderr",
        description="Append a message to the process stderr buffer.",
        param_descriptions={},
        requires_context=True,
    )
    async def write_stderr_tool(self, message: str, runtime_context: Optional[dict[str, Any]] = None) -> ToolResult:
        """Append a message to the log and trigger callbacks."""
        if not message or not isinstance(message, str):
            return ToolResult.from_error("message must be a non-empty string")

        self.log.append(message)
        return ToolResult.from_success(message)

    def hook_provide_tools(self) -> list:
        """Return the stderr tool."""
        return apply_tool_overrides([self.write_stderr_tool], self.config.tools)
