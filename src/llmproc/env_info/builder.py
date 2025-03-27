"""Environment information builder for LLM programs."""

import datetime
import getpass
import os
import platform
from pathlib import Path


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
    def build_preload_content(preloaded_content: dict) -> str:
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
        base_prompt: str, env_config: dict, preloaded_content: dict = None, 
        include_env: bool = True, file_descriptor_enabled: bool = False,
        references_enabled: bool = False, page_user_input: bool = False
    ) -> str:
        """Get enhanced system prompt with preloaded files and environment info.

        Args:
            base_prompt: Base system prompt
            env_config: Environment configuration dictionary
            preloaded_content: Dictionary mapping file paths to content
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
            from llmproc.tools import file_descriptor_base_instructions
            parts.append(file_descriptor_base_instructions)
            
            # Add user input paging instructions if enabled
            if page_user_input:
                from llmproc.tools import fd_user_input_instructions
                parts.append(fd_user_input_instructions)
            
        # Add reference instructions if enabled
        if references_enabled and file_descriptor_enabled:
            from llmproc.tools import reference_instructions
            parts.append(reference_instructions)

        # Add preloaded content if available
        if preloaded_content:
            preload_content = EnvInfoBuilder.build_preload_content(preloaded_content)
            if preload_content:
                parts.append(preload_content)

        # Combine all parts with proper spacing
        return "\n\n".join(parts)
