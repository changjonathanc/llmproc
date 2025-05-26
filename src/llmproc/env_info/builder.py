"""Environment information builder for LLM programs."""

import datetime
import getpass
import logging
import os
import platform
import warnings
from pathlib import Path
from typing import Any, Optional

from llmproc.tools import (
    fd_user_input_instructions,
    file_descriptor_instructions,
    reference_instructions,
)

logger = logging.getLogger(__name__)


class EnvInfoBuilder:
    """Builder for environment information in system prompts."""

    @staticmethod
    def build_env_info(env_config: dict, include_env: bool = True) -> str:
        """Build environment information string based on configuration.

        Args:
            env_config: Environment configuration dictionary
            include_env: Whether to include environment info

        Returns:
            Formatted environment information string
        """
        # Skip if environment info is disabled
        if not include_env:
            return ""

        variables = env_config.get("variables", [])

        # Skip if no variables are specified
        if not variables:
            return ""

        # Start the env section
        env_info = "<env>\n"

        # Handle standard variables based on the requested list or "all"
        all_variables = variables == "all"
        var_list = (
            [
                "working_directory",
                "platform",
                "date",
                "python_version",
                "hostname",
                "username",
            ]
            if all_variables
            else variables
        )

        # Add standard environment information if requested
        if "working_directory" in var_list:
            env_info += f"working_directory: {os.getcwd()}\n"

        if "platform" in var_list:
            env_info += f"platform: {platform.system().lower()}\n"

        if "date" in var_list:
            env_info += f"date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n"

        if "python_version" in var_list:
            env_info += f"python_version: {platform.python_version()}\n"

        if "hostname" in var_list:
            env_info += f"hostname: {platform.node()}\n"

        if "username" in var_list:
            env_info += f"username: {getpass.getuser()}\n"

        # Add any custom environment variables
        for key, value in env_config.items():
            # Skip the variables key and any non-string values
            if key == "variables" or not isinstance(value, str):
                continue
            env_info += f"{key}: {value}\n"

        # Close the env section
        env_info += "</env>"

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
            # Resolve relative paths against the base directory
            path = Path(file_path_str)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            else:
                path = path.resolve()  # Resolve absolute paths too

            try:
                if not path.exists():
                    EnvInfoBuilder._warn_preload("Preload file not found", file_path_str, path)
                    continue  # Skip missing files

                if not path.is_file():
                    EnvInfoBuilder._warn_preload("Preload path is not a file", file_path_str, path)
                    continue  # Skip non-files

                content = path.read_text()
                content_dict[str(path)] = content
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

        preload_content = "<preload>\n"
        for file_path, content in preloaded_content.items():
            filename = Path(file_path).name
            preload_content += f'<file path="{filename}">\n{content}\n</file>\n'
        preload_content += "</preload>"

        return preload_content

    @staticmethod
    def get_enriched_system_prompt(
        base_prompt: str,
        env_config: dict,
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
            env_config: Environment configuration dictionary
            preloaded_content: Dictionary mapping file paths to content (deprecated, prefer preload_files)
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
