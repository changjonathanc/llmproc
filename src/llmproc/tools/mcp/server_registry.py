"""MCP server configuration utilities."""

from __future__ import annotations

from pydantic import BaseModel


class MCPServerSettings(BaseModel):
    """Configuration settings for an individual MCP server."""

    type: str = "stdio"  # "stdio" or "sse"
    command: str | None = None
    args: list[str] | None = None
    url: str | None = None
    env: dict | None = None
    description: str | None = None

    @property
    def transport(self) -> str:
        return self.type

    @transport.setter
    def transport(self, value: str) -> None:
        self.type = value
