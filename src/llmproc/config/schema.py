"""Configuration schema for LLM programs using Pydantic models."""

import warnings
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

from llmproc.common.access_control import AccessLevel

# Import the simplified MCP models
from llmproc.config.mcp import MCPServerTools, MCPToolsConfig
from llmproc.config.tool import ToolConfig
from llmproc.config.utils import resolve_path
from llmproc.env_info.constants import STANDARD_VAR_NAMES


class ModelConfig(BaseModel):
    """Model configuration section."""

    name: str
    provider: str
    disable_automatic_caching: bool = False
    project_id: str | None = None
    region: str | None = None
    max_iterations: int = 10

    @classmethod
    @field_validator("provider")
    def validate_provider(cls, v):
        """Validate that the provider is supported."""
        supported_providers = {"openai", "anthropic", "vertex"}
        if v not in supported_providers:
            raise ValueError(f"Provider '{v}' not supported. Must be one of: {', '.join(supported_providers)}")
        return v

    @model_validator(mode="after")
    def validate_caching_config(self):
        """Validate that automatic caching is only disabled for Anthropic providers."""
        if self.disable_automatic_caching and "anthropic" not in self.provider:
            warnings.warn(
                f"'disable_automatic_caching' is set to true, but the provider is '{self.provider}'. Automatic caching is only supported for Anthropic providers, so this setting will have no effect.",
                stacklevel=2,
            )
        return self


class PromptConfig(BaseModel):
    """Prompt configuration section."""

    model_config = {"populate_by_name": True}

    system_prompt: str | None = Field(default="", alias="system")
    system_prompt_file: str | None = None
    user: str | None = Field(default=None, alias="user_prompt")

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
        # First check for system_prompt_file (takes precedence)
        if self.system_prompt_file:
            try:
                file_path = resolve_path(
                    self.system_prompt_file,
                    base_dir,
                    must_exist=True,
                    error_prefix="System prompt file",
                )
                return file_path.read_text()
            except FileNotFoundError as e:
                # Re-raise the error with the same message
                raise FileNotFoundError(str(e))

        # Return system_prompt (or empty string if neither is specified)
        return self.system_prompt or ""


class PreloadConfig(BaseModel):
    """Preload configuration section."""

    files: list[str] = []
    relative_to: Literal["program", "cwd"] = "program"


class MCPConfig(BaseModel):
    """MCP configuration section."""

    config_path: str | None = None
    servers: dict[str, dict] | None = None
    # tools field has been moved to ToolsConfig.mcp


class ToolsConfig(BaseModel):
    """Tools configuration section."""

    builtin: list[str | ToolConfig] = Field(default_factory=list, alias="enabled")
    mcp: MCPToolsConfig | None = None  # MCP tools configuration moved from [mcp.tools]

    model_config = {"populate_by_name": True}


class EnvInfoConfig(BaseModel):
    """Environment information configuration section."""

    variables: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    env_vars: dict[str, str] = Field(default_factory=dict)
    file_map_root: str | None = None
    file_map_max_files: int = 50
    file_map_show_size: bool = True
    # Allow additional custom environment variables as strings
    model_config = {"extra": "allow"}

    @field_validator("variables", mode="before")
    @classmethod
    def parse_variables(cls, v):
        """Normalize variables field input."""
        if v == "all":
            return STANDARD_VAR_NAMES
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("file_map_max_files")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate that integer values are positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class FileDescriptorConfig(BaseModel):
    """File descriptor configuration section."""

    enabled: bool = False
    max_direct_output_chars: int = 8000
    default_page_size: int = 4000
    max_input_chars: int = 8000
    page_user_input: bool = True
    enable_references: bool = False

    @classmethod
    @field_validator("max_direct_output_chars", "default_page_size", "max_input_chars")
    def validate_positive_int(cls, v):
        """Validate that integer values are positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class DemoConfig(BaseModel):
    """Demo configuration for multi-turn demonstrations."""

    prompts: list[str] = []
    pause_between_prompts: bool = True
    display_name: str | None = None


class LinkedProgramItem(BaseModel):
    """Configuration for a single linked program."""

    path: str
    description: str = ""


class LinkedProgramsConfig(RootModel):
    """Linked programs configuration section."""

    root: dict[str, str | LinkedProgramItem] = {}


class LLMProgramConfig(BaseModel):
    """Full LLM program configuration."""

    model: ModelConfig
    prompt: PromptConfig = PromptConfig()
    parameters: dict[str, Any] = {}
    preload: PreloadConfig | None = PreloadConfig()
    mcp: MCPConfig | None = None
    tools: ToolsConfig | None = ToolsConfig()
    env_info: EnvInfoConfig | None = EnvInfoConfig()
    linked_programs: LinkedProgramsConfig | None = LinkedProgramsConfig()
    file_descriptor: FileDescriptorConfig | None = None
    demo: DemoConfig | None = None

    model_config = {
        "extra": "forbid"  # Forbid extra fields
    }

    @field_validator("parameters")
    @classmethod
    def validate_reasoning_parameters(cls, v):
        """Validate that reasoning parameters have valid values."""
        # Check reasoning_effort values
        if "reasoning_effort" in v:
            valid_values = {"low", "medium", "high"}
            if v["reasoning_effort"] not in valid_values:
                raise ValueError(
                    f"Invalid reasoning_effort value '{v['reasoning_effort']}'. Must be one of: {', '.join(valid_values)}"
                )

        # Check for token parameter conflicts
        if "max_tokens" in v and "max_completion_tokens" in v:
            raise ValueError(
                "Cannot specify both 'max_tokens' and 'max_completion_tokens'. Use 'max_tokens' for standard models and 'max_completion_tokens' for reasoning models."
            )

        # Validate extra_headers structure if present
        if "extra_headers" in v and not isinstance(v["extra_headers"], dict):
            raise ValueError("parameters.extra_headers must be a dictionary of header key-value pairs")

        # Validate thinking structure if present
        if "thinking" in v:
            thinking = v["thinking"]
            if not isinstance(thinking, dict):
                raise ValueError("parameters.thinking must be a dictionary")

            # Check for required fields
            if "type" in thinking and thinking["type"] not in ["enabled", "disabled"]:
                raise ValueError("parameters.thinking.type must be 'enabled' or 'disabled'")

            # Check budget_tokens if present
            if "budget_tokens" in thinking:
                budget = thinking["budget_tokens"]
                if not isinstance(budget, int):
                    raise ValueError("parameters.thinking.budget_tokens must be an integer")
                if budget < 0:
                    raise ValueError("parameters.thinking.budget_tokens must be non-negative")
                if budget > 0 and budget < 1024:
                    warnings.warn(
                        f"parameters.thinking.budget_tokens set to {budget}, but Claude requires minimum 1024. This will likely fail at runtime.",
                        stacklevel=2,
                    )

        return v

    @model_validator(mode="after")
    def validate_file_descriptor(self):
        """Validate file descriptor configuration is consistent with tools.

        This validator checks if ``file_descriptor.enabled`` is true but no FD tools are enabled.
        It also issues a warning if there's a ``file_descriptor`` section but no FD tools.
        """
        # FD tools
        fd_tools = ["read_fd", "fd_to_file"]

        # Check if file_descriptor is configured
        if self.file_descriptor:
            # Check if any FD tools are enabled
            has_fd_tools = False
            if self.tools and self.tools.builtin:
                has_fd_tools = any((t if isinstance(t, str) else t.name) in fd_tools for t in self.tools.builtin)

            # If explicitly enabled but no tools, raise error
            if self.file_descriptor.enabled and not has_fd_tools:
                raise ValueError(
                    "file_descriptor.enabled is set to true, but no file descriptor tools "
                    "('read_fd', 'fd_to_file') are enabled in the [tools] section. "
                    "Add at least 'read_fd' to the enabled tools list."
                )

            # If has settings but not explicitly enabled and no tools, issue warning
            if not self.file_descriptor.enabled and not has_fd_tools:
                warnings.warn(
                    "File descriptor configuration is present but no file descriptor tools "
                    "are enabled in the [tools] section and file_descriptor.enabled is not true. "
                    "The configuration will have no effect.",
                    stacklevel=2,
                )

        return self

    @model_validator(mode="after")
    def validate_parameters(self):
        """Validate the parameters dictionary and issue warnings for unknown parameters.

        This validator doesn't reject unknown parameters, it just issues warnings.
        We want to stay flexible as LLM APIs evolve, but provide guidance on what's expected.
        """
        # Standard LLM API parameters that we expect to see
        known_parameters = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            # OpenAI specific
            "top_k",
            "stop",
            "reasoning_effort",  # For OpenAI reasoning models
            "max_completion_tokens",  # For OpenAI reasoning models (replaces max_tokens)
            # Anthropic specific
            "max_tokens_to_sample",
            "stop_sequences",
            "thinking",  # For Claude 3.7+ thinking models
            "extra_headers",  # For API-specific headers like beta features
        }

        if self.parameters:
            for param_name in self.parameters:
                if param_name not in known_parameters:
                    warnings.warn(
                        f"Unknown API parameter '{param_name}' in configuration. This may be a typo or a newer parameter not yet recognized.",
                        stacklevel=2,
                    )

            # Check if using OpenAI reasoning model - in this case we need special parameter handling
            is_reasoning_model = False
            if hasattr(self, "model") and self.model.provider == "openai":
                is_reasoning_model = self.model.name.startswith(("o1", "o3"))

                # If using a reasoning model, suggest recommended parameters
                if is_reasoning_model and "reasoning_effort" not in self.parameters:
                    warnings.warn(
                        "OpenAI reasoning model detected (o1, o3). For better results, consider adding the 'reasoning_effort' parameter (low, medium, high).",
                        stacklevel=2,
                    )

            # Check if using Claude 3.7 thinking model
            is_claude_thinking_model = False
            if hasattr(self, "model") and self.model.provider == "anthropic":
                is_claude_thinking_model = self.model.name.startswith("claude-3-7")

                # Removing the Claude 3.7+ thinking model warning for better testing experience
                # This was causing noise in test output and is not critical for functionality
                pass

            # Check if reasoning_effort used with non-OpenAI provider
            if "reasoning_effort" in self.parameters and hasattr(self, "model") and self.model.provider != "openai":
                warnings.warn(
                    "The 'reasoning_effort' parameter is only supported with OpenAI reasoning models. It will be ignored for other providers.",
                    stacklevel=2,
                )

            # Check if thinking parameters used with non-Claude provider or non-3.7 Claude
            if (
                "thinking" in self.parameters
                and hasattr(self, "model")
                and (self.model.provider != "anthropic" or not self.model.name.startswith("claude-3-7"))
            ):
                warnings.warn(
                    "The 'thinking' parameter is only supported with Claude 3.7+ models. It will be ignored for other providers.",
                    stacklevel=2,
                )

            # Validate that reasoning models use max_completion_tokens
            if hasattr(self, "model") and self.model.provider == "openai":
                if is_reasoning_model and "max_tokens" in self.parameters:
                    warnings.warn(
                        "OpenAI reasoning models (o1, o3) should use 'max_completion_tokens' instead of 'max_tokens'. Your configuration may fail at runtime.",
                        stacklevel=2,
                    )
                elif not is_reasoning_model and "max_completion_tokens" in self.parameters:
                    warnings.warn(
                        "'max_completion_tokens' is only for OpenAI reasoning models (o1, o3). Standard models should use 'max_tokens' instead.",
                        stacklevel=2,
                    )

        return self

    def get_api_parameters(self) -> dict[str, Any]:
        """Extract API parameters from the parameters dictionary.

        This method filters the parameters to only include those that are relevant
        to the LLM API calls. Unlike the _extract_api_parameters method in LLMProgram,
        this does NOT filter out unknown parameters, maintaining flexibility.

        Returns:
            Dictionary of parameters to pass to the LLM API
        """
        # For now, we're being permissive and returning all parameters
        # This allows for flexibility as APIs evolve
        return self.parameters.copy() if self.parameters else {}
