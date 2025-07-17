from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, kw_only=True)
class ProgramConfigData:
    """Container for :class:`LLMProgram` initialization attributes."""

    model_name: str
    provider: str
    system_prompt: str | None = None
    parameters: dict[str, Any] | None = None
    mcp_config_path: str | None = None
    mcp_servers: dict[str, dict] | None = None
    tools: list[Any] | None = None
    tools_config: Any | None = None  # Original ToolsConfig object with official anthropic field
    plugins: list[Any] | None = None
    plugin_configs: dict[str, Any] | None = None
    base_dir: Path | None = None
    project_id: str | None = None
    region: str | None = None
    user_prompt: str | None = None
    max_iterations: int = 10
