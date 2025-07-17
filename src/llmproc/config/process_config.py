"""Dataclass for constructing :class:`~llmproc.llm_process.LLMProcess`.

This configuration object collects all parameters required by
``LLMProcess.__init__`` in a structured form. It mirrors the existing
constructor arguments but allows for centralized validation and easier
extension.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmproc.common.access_control import AccessLevel
from llmproc.tools import ToolManager

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    # Imported here to avoid circular dependency with LLMProgram
    from llmproc.program import LLMProgram


@dataclass(slots=True, kw_only=True)
class ProcessConfig:
    """Container for all ``LLMProcess`` initialization attributes."""

    program: LLMProgram
    model_name: str
    provider: str
    base_system_prompt: str
    access_level: AccessLevel = AccessLevel.ADMIN
    base_dir: Path | None = None
    api_params: dict[str, Any] = field(default_factory=dict)
    state: list[dict[str, Any]] = field(default_factory=list)
    enriched_system_prompt: str | None = None
    client: Any = None
    tool_manager: ToolManager | None = None
    mcp_config_path: str | None = None
    mcp_servers: dict[str, Any] | None = None
    mcp_tools: dict[str, Any] = field(default_factory=dict)
    mcp_enabled: bool | None = None
    user_prompt: str | None = None
    max_iterations: int = 10
    plugins: list[Any] = field(default_factory=list)
    loop: asyncio.AbstractEventLoop | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.model_name or not self.provider:
            raise ValueError("model_name and provider are required")

        # File descriptor validation is now handled by the FileDescriptorPlugin
