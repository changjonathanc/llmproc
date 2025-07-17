"""Base classes for provider-hosted server tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llmproc.common.metadata import ToolMeta
from llmproc.common.results import ToolResult
from llmproc.tools.core import Tool


class ServerTool(Tool, ABC):
    """Base class for tools executed on provider servers."""

    def __init__(self, name: str, provider: str, config: dict[str, Any], **kwargs: Any) -> None:
        self.provider = provider
        self.provider_config = config
        super().__init__(
            handler=None,
            schema=self.to_api_definition(),
            meta=self._create_meta(name),
            **kwargs,
        )

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
        process_access_level=None,
    ) -> ToolResult:
        """Server tools cannot execute locally."""
        raise NotImplementedError(
            f"Server tool '{self.meta.name}' executes on {self.provider} servers, not locally via ToolManager"
        )

    @abstractmethod
    def to_api_definition(self) -> dict[str, Any]:
        """Convert to provider API tool definition."""

    @abstractmethod
    def _create_meta(self, name: str) -> ToolMeta:
        """Create tool metadata."""
