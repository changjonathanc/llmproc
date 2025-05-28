"""Environment information builder for LLM programs."""

import logging
import os
import subprocess
import warnings
from pathlib import Path
from typing import Optional

from llmproc.config import EnvInfoConfig
from llmproc.env_info.constants import STANDARD_VAR_FUNCTIONS
from llmproc.tools import (
    fd_user_input_instructions,
    file_descriptor_instructions,
    reference_instructions,
)

logger = logging.getLogger(__name__)


def _sanitize_command(cmd: str) -> str:
    """Sanitize a shell command name for use as a tag."""
    return cmd.replace(" ", "_").replace("/", "_").replace("\t", "_")


class EnvInfoBuilder:
    """Builder for environment information in system prompts."""

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _build_file_map_lines(env_config: EnvInfoConfig) -> list[str]:
        """Build file map lines for the environment info block."""
        root_dir = Path(env_config.file_map_root or ".").resolve()
        show_size = bool(env_config.file_map_show_size)
        max_files = int(env_config.file_map_max_files)
        if not root_dir.exists() or not root_dir.is_dir():
            warnings.warn(
                f"file_map_root does not exist or is not a directory: {root_dir}",
                stacklevel=2,
            )
            return []

        files = [p for p in root_dir.rglob("*") if p.is_file()]
        lines = ["file_map:"]
        for path in files[:max_files]:
            rel = path.relative_to(root_dir)
            size_part = ""
            if show_size:
                try:
                    size_part = f" ({path.stat().st_size} bytes)"
                except OSError:
                    size_part = ""
            lines.append(f"  {rel}{size_part}")
        if len(files) > max_files:
            lines.append(f"  ... ({len(files) - max_files} more files)")
        return lines

    @staticmethod
    def _build_variable_lines(variables: list[str], env_config: EnvInfoConfig) -> list[str]:
        """Build lines for standard variables."""
        lines: list[str] = []
        for var in variables:
            if var in STANDARD_VAR_FUNCTIONS:
                lines.append(f"{var}: {STANDARD_VAR_FUNCTIONS[var]()}")
            elif var == "file_map":
                lines.extend(EnvInfoBuilder._build_file_map_lines(env_config))
        return lines

    @staticmethod
    def _build_custom_var_lines(env_config: EnvInfoConfig) -> list[str]:
        """Build lines for custom variables in the config."""
        return [f"{k}: {v}" for k, v in (env_config.model_extra or {}).items() if isinstance(v, str)]

    @staticmethod
    def _build_env_var_lines(env_config: EnvInfoConfig) -> list[str]:
        """Build lines for values pulled from actual environment variables."""
        lines = []
        for label, var_name in env_config.env_vars.items():
            env_value = os.getenv(var_name)
            if env_value:
                lines.append(f"{label}: {env_value.rstrip()}")
        return lines

    @staticmethod
    def _build_command_lines(commands: list[str]) -> list[str]:
        """Build lines for command outputs."""
        lines: list[str] = []
        for i, cmd in enumerate(commands):
            lines.append(f"> {cmd}")
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                output = result.stdout.strip()
                if output:
                    lines.extend(output.splitlines())
                if result.returncode != 0:
                    lines.append(f"error({result.returncode})")
            except Exception as e:  # Catch unexpected errors
                logger.warning("Unexpected error running env_info command '%s': %s", cmd, e)
                lines.append("error")
            if i < len(commands) - 1:
                lines.append("")
        return lines

    @staticmethod
    def _resolve_path(base_dir: Path, file_path_str: str) -> Path:
        """Resolve ``file_path_str`` relative to ``base_dir``."""
        path = Path(file_path_str)
        if not path.is_absolute():
            return (base_dir / path).resolve()
        return path.resolve()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def build_env_info(env_config: EnvInfoConfig, include_env: bool = True) -> str:
        """Build environment information string from ``EnvInfoConfig``."""
        if not include_env:
            return ""

        if not env_config.variables and not env_config.commands:
            return ""

        lines = (
            EnvInfoBuilder._build_variable_lines(env_config.variables, env_config)
            + EnvInfoBuilder._build_custom_var_lines(env_config)
            + EnvInfoBuilder._build_env_var_lines(env_config)
            + EnvInfoBuilder._build_command_lines(env_config.commands)
        )

        if not lines:
            return ""

        env_info = "<env>\n" + "\n".join(lines) + "\n</env>"
        return env_info

    @staticmethod
    def _warn_preload(
        message: str,
        specified_path: str,
        resolved_path: Path,
        error: Optional[Exception] = None,
    ):
        """Helper to issue consistent warnings for preloading issues.

        Args:
            message: Base warning message
            specified_path: The path as specified in the configuration
            resolved_path: The resolved path
            error: Optional exception that occurred
        """
        full_message = f"{message} - Specified: '{specified_path}', Resolved: '{resolved_path}'"
        if error:
            full_message += f", Error: {error}"
        warnings.warn(full_message, stacklevel=3)

    @staticmethod
    def load_files(file_paths: list[str], base_dir: Optional[Path] = None) -> dict[str, str]:
        """Load content from multiple files.

        Args:
            file_paths: List of file paths to load
            base_dir: Base directory for resolving relative paths, defaults to current directory

        Returns:
            Dictionary mapping file paths to their content
        """
        if not file_paths:
            return {}

        base_dir = base_dir or Path(".")
        content_dict = {}

        for file_path_str in file_paths:
            path = EnvInfoBuilder._resolve_path(base_dir, file_path_str)
            try:
                if not path.exists():
                    EnvInfoBuilder._warn_preload("Preload file not found", file_path_str, path)
                    continue

                if not path.is_file():
                    EnvInfoBuilder._warn_preload("Preload path is not a file", file_path_str, path)
                    continue

                content_dict[str(path)] = path.read_text()
                logger.debug(f"Successfully preloaded content from: {path}")

            except OSError as e:
                EnvInfoBuilder._warn_preload("Error reading preload file", file_path_str, path, e)
            except Exception as e:  # Catch other potential errors
                EnvInfoBuilder._warn_preload("Unexpected error preloading file", file_path_str, path, e)

        return content_dict

    @staticmethod
    def build_preload_content(preloaded_content: dict[str, str]) -> str:
        """Build preloaded content string.

        Args:
            preloaded_content: Dictionary mapping file paths to content

        Returns:
            Formatted preloaded content string
        """
        if not preloaded_content:
            return ""

        files = [
            f'<file path="{Path(file_path).name}">\n{content}\n</file>'
            for file_path, content in preloaded_content.items()
        ]
        return "<preload>\n" + "\n".join(files) + "\n</preload>"

    @staticmethod
    def get_enriched_system_prompt(
        base_prompt: str,
        env_config: EnvInfoConfig,
        preloaded_content: Optional[dict[str, str]] = None,
        preload_files: Optional[list[str]] = None,
        base_dir: Optional[Path] = None,
        include_env: bool = True,
        file_descriptor_enabled: bool = False,
        references_enabled: bool = False,
        page_user_input: bool = False,
    ) -> str:
        """Get enhanced system prompt with preloaded files and environment info.

        Args:
            base_prompt: Base system prompt
            env_config: Environment configuration object
            preloaded_content: Dictionary mapping file paths to content (deprecated, prefer ``preload_files``)
            preload_files: List of file paths to preload
            base_dir: Base directory for resolving relative paths in preload_files
            include_env: Whether to include environment information
            file_descriptor_enabled: Whether file descriptor system is enabled
            references_enabled: Whether reference ID system is enabled
            page_user_input: Whether user input paging is enabled

        Returns:
            Complete system prompt ready for API calls
        """
        # Start with the base system prompt
        parts = [base_prompt]

        # Add environment info if configured
        env_info = EnvInfoBuilder.build_env_info(env_config, include_env)
        if env_info:
            parts.append(env_info)

        # Add file descriptor instructions if enabled
        if file_descriptor_enabled:
            parts.append(file_descriptor_instructions)

            # Add user input paging instructions if enabled
            if page_user_input:
                parts.append(fd_user_input_instructions)

        # Add reference instructions if enabled
        if references_enabled and file_descriptor_enabled:
            parts.append(reference_instructions)

        # Handle preloaded content
        combined_content = {}

        # Load files if file paths are provided
        if preload_files:
            file_content = EnvInfoBuilder.load_files(preload_files, base_dir)
            combined_content.update(file_content)

        # Also support direct preloaded content for backward compatibility
        if preloaded_content:
            combined_content.update(preloaded_content)

        # Add preloaded content if available
        if combined_content:
            preload_content = EnvInfoBuilder.build_preload_content(combined_content)
            if preload_content:
                parts.append(preload_content)

        # Combine all parts with proper spacing
        return "\n\n".join(parts)
