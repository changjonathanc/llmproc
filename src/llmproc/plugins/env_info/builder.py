"""Environment information builder for LLM programs."""

import logging

from llmproc.config.schema import EnvInfoConfig

from .constants import STANDARD_VAR_FUNCTIONS

logger = logging.getLogger(__name__)


class EnvInfoBuilder:
    """Builder for environment information in system prompts."""

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _build_variable_lines(variables: list[str], env_config: EnvInfoConfig) -> list[str]:
        """Build lines for standard variables."""
        lines: list[str] = []
        for var in variables:
            if var in STANDARD_VAR_FUNCTIONS:
                lines.append(f"{var}: {STANDARD_VAR_FUNCTIONS[var]()}")
        return lines

    @staticmethod
    def _build_custom_var_lines(env_config: EnvInfoConfig) -> list[str]:
        """Build lines for custom variables in the config."""
        return [f"{k}: {v}" for k, v in (env_config.model_extra or {}).items() if isinstance(v, str)]

    @staticmethod
    # _build_env_var_lines and _build_command_lines removed; use plugins instead

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def build_env_info(env_config: EnvInfoConfig, include_env: bool = True) -> str:
        """Build environment information string from ``EnvInfoConfig``."""
        if not include_env:
            return ""

        if not env_config.variables:
            return ""

        lines = EnvInfoBuilder._build_variable_lines(
            env_config.variables, env_config
        ) + EnvInfoBuilder._build_custom_var_lines(env_config)

        if not lines:
            return ""

        env_info = "<env>\n" + "\n".join(lines) + "\n</env>"
        return env_info
