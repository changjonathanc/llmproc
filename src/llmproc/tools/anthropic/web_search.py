"""Anthropic web search server tool implementation."""

from __future__ import annotations

from typing import Any

from llmproc.common.access_control import AccessLevel
from llmproc.common.metadata import ToolMeta
from llmproc.config.schema import AnthropicWebSearchConfig
from llmproc.tools.server_tool import ServerTool


class WebSearchTool(ServerTool):
    """Anthropic web search server-side tool."""

    config: AnthropicWebSearchConfig

    def __init__(self, config_dict: dict[str, Any]):
        self.config = AnthropicWebSearchConfig.model_validate(config_dict)
        super().__init__(
            name="web_search",
            provider="anthropic",
            config=config_dict,
        )

    def to_api_definition(self) -> dict[str, Any]:
        """Convert to Anthropic API tool definition."""
        return {
            "type": "web_search_20250305",
            "name": "web_search",
            **self.config.model_dump(exclude={"enabled"}, exclude_none=True),
        }

    def _create_meta(self, name: str) -> ToolMeta:
        return ToolMeta(
            name=name,
            description="Web search via Anthropic servers",
            access=AccessLevel.READ,
        )
