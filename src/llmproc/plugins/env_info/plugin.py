from __future__ import annotations

import logging

from llmproc.config.schema import EnvInfoConfig

from .builder import EnvInfoBuilder

logger = logging.getLogger(__name__)


class EnvInfoPlugin:
    """Plugin that appends environment information to the system prompt."""

    def __init__(self, env_config: EnvInfoConfig) -> None:
        self.env_config = env_config

    async def hook_system_prompt(self, system_prompt: str, process) -> str | None:
        env_text = EnvInfoBuilder.build_env_info(self.env_config)
        if env_text:
            return f"{system_prompt}\n\n{env_text}"
        return None


__all__ = ["EnvInfoPlugin", "EnvInfoBuilder"]
