"""Configuration schema for LLM programs using Pydantic models."""

from typing import Any, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    RootModel,
    ValidationError,
    field_validator,
    model_validator,
)


class ModelConfig(BaseModel):
    """Model configuration section."""

    name: str
    provider: str
    display_name: str | None = None

    @classmethod
    @field_validator("provider")
    def validate_provider(cls, v):
        """Validate that the provider is supported."""
        supported_providers = {"openai", "anthropic", "vertex"}
        if v not in supported_providers:
            raise ValueError(
                f"Provider '{v}' not supported. Must be one of: {', '.join(supported_providers)}"
            )
        return v


class PromptConfig(BaseModel):
    """Prompt configuration section."""

    system_prompt: str | None = ""
    system_prompt_file: str | None = None

    @model_validator(mode="after")
    def check_prompt_sources(self):
        """Check that at least one prompt source is provided."""
        if not self.system_prompt and not self.system_prompt_file:
            # Set default empty system prompt
            self.system_prompt = ""

        return self

    def resolve(self, base_dir=None):
        """Resolve the system prompt, loading from file if specified.

        Args:
            base_dir: Base directory for resolving relative file paths

        Returns:
            Resolved system prompt string

        Raises:
            FileNotFoundError: If system_prompt_file is specified but doesn't exist
        """
        import os
        from pathlib import Path

        # First check for system_prompt_file (takes precedence)
        if self.system_prompt_file:
            # Determine file path, using base_dir for relative paths if provided
            file_path = Path(self.system_prompt_file)
            if not file_path.is_absolute() and base_dir:
                file_path = base_dir / file_path

            if file_path.exists():
                return file_path.read_text()
            else:
                raise FileNotFoundError(
                    f"System prompt file not found - Specified: '{self.system_prompt_file}', Resolved: '{file_path}'"
                )

        # Return system_prompt (or empty string if neither is specified)
        return self.system_prompt or ""


class PreloadConfig(BaseModel):
    """Preload configuration section."""

    files: list[str] = []


class MCPToolsConfig(RootModel):
    """MCP tools configuration."""

    root: dict[str, list[str] | str] = {}

    @classmethod
    @field_validator("root")
    def validate_tools(cls, v):
        """Validate that tool configurations are either lists or 'all'."""
        for server, tools in v.items():
            if not isinstance(tools, list) and tools != "all":
                raise ValueError(
                    f"Tool configuration for server '{server}' must be 'all' or a list of tool names"
                )
        return v


class MCPConfig(BaseModel):
    """MCP configuration section."""

    config_path: str | None = None
    tools: MCPToolsConfig | None = None


class ToolsConfig(BaseModel):
    """Tools configuration section."""

    enabled: list[str] = []


class EnvInfoConfig(BaseModel):
    """Environment information configuration section."""

    variables: list[str] | str = []  # Empty list by default (disabled)
    # Allow additional custom environment variables as strings
    model_config = {"extra": "allow"}


class DebugConfig(BaseModel):
    """Debug configuration section."""

    debug_tools: bool = False


class LinkedProgramsConfig(RootModel):
    """Linked programs configuration section."""

    root: dict[str, str] = {}


class LLMProgramConfig(BaseModel):
    """Full LLM program configuration."""

    model: ModelConfig
    prompt: PromptConfig = PromptConfig()
    parameters: dict[str, Any] = {}
    preload: PreloadConfig | None = PreloadConfig()
    mcp: MCPConfig | None = None
    tools: ToolsConfig | None = ToolsConfig()
    env_info: EnvInfoConfig | None = EnvInfoConfig()
    debug: DebugConfig | None = DebugConfig()
    linked_programs: LinkedProgramsConfig | None = LinkedProgramsConfig()

    model_config = {
        "extra": "forbid"  # Forbid extra fields
    }