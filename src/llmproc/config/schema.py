"""Configuration schema for LLM programs using Pydantic models."""

import warnings
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

# Import the simplified MCP models
from llmproc.config.mcp import MCPToolsConfig
from llmproc.config.tool import ToolConfig
from llmproc.config.utils import resolve_path
from llmproc.plugins.env_info.constants import STANDARD_VAR_NAMES


class ModelConfig(BaseModel):
    """Model configuration section."""

    name: str
    provider: str
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


class PreloadFilesPluginConfig(BaseModel):
    """Configuration for the preload files plugin."""

    files: list[str] = Field(default_factory=list)
    relative_to: Literal["program", "cwd"] = "program"


class MCPConfig(BaseModel):
    """MCP configuration section."""

    config_path: str | None = None
    servers: dict[str, dict] | None = None
    # tools field has been moved to ToolsConfig.mcp


class AnthropicWebSearchConfig(BaseModel):
    """Anthropic web search tool configuration."""

    enabled: bool = False
    max_uses: int | None = Field(None, ge=1, le=50, description="Max searches per request")
    allowed_domains: list[str] | None = Field(None, description="Whitelist of allowed domains")
    blocked_domains: list[str] | None = Field(None, description="Blacklist of blocked domains")
    user_location: dict[str, Any] | None = None


class AnthropicToolsConfig(BaseModel):
    """Anthropic provider-specific tools configuration."""

    web_search: AnthropicWebSearchConfig | None = None


class OpenAIWebSearchConfig(BaseModel):
    """OpenAI web search tool configuration."""

    enabled: bool = False
    search_context_size: str = Field("medium", description="How much context to include in search (low/medium/high)")
    user_location: dict[str, Any] | None = None


class OpenAIToolsConfig(BaseModel):
    """OpenAI provider-specific tools configuration."""

    web_search: OpenAIWebSearchConfig | None = None


class ToolsConfig(BaseModel):
    """Tools configuration section."""

    builtin: list[str | ToolConfig] = Field(default_factory=list)
    mcp: MCPToolsConfig | None = None  # MCP tools configuration moved from [mcp.tools]
    anthropic: AnthropicToolsConfig | None = None  # Anthropic provider tools
    openai: OpenAIToolsConfig | None = None  # OpenAI provider tools

    model_config = {"populate_by_name": True}


class EnvInfoConfig(BaseModel):
    """Environment information configuration section."""

    variables: list[str] = Field(default_factory=list)
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


# Legacy FileDescriptorConfig removed - use plugins.file_descriptor instead


class FileDescriptorPluginConfig(BaseModel):
    """Configuration for the file descriptor plugin."""

    max_direct_output_chars: int = 8000
    default_page_size: int = 4000
    max_input_chars: int = 8000
    page_user_input: bool = True
    enable_references: bool = False
    tools: list[str | ToolConfig] = Field(default_factory=list)

    @classmethod
    @field_validator(
        "max_direct_output_chars",
        "default_page_size",
        "max_input_chars",
    )
    def validate_positive_int(cls, v):
        """Validate that integer values are positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class MessageIDPluginConfig(BaseModel):
    """Configuration for the message ID plugin."""

    enable_goto: bool = True
    tools: list[str | ToolConfig] = Field(default_factory=list)


class SpawnPluginConfig(BaseModel):
    """Configuration for the spawn plugin."""

    linked_programs: dict[str, str] = Field(default_factory=dict)
    linked_program_descriptions: dict[str, str] = Field(default_factory=dict)
    tools: list[str | ToolConfig] = Field(default_factory=list)


class StderrPluginConfig(BaseModel):
    """Configuration for the stderr logging plugin."""

    tools: list[str | ToolConfig] = Field(default_factory=list)


class EnvInfoPluginConfig(EnvInfoConfig):
    """Configuration for the environment info plugin."""

    pass


class PluginsConfig(BaseModel):
    """Root model for plugin configurations."""

    file_descriptor: FileDescriptorPluginConfig | None = None
    message_id: MessageIDPluginConfig | None = None
    spawn: SpawnPluginConfig | None = None
    stderr: StderrPluginConfig | None = None
    preload_files: PreloadFilesPluginConfig | None = None
    env_info: EnvInfoPluginConfig | None = None

    model_config = {"extra": "allow"}


class DemoConfig(BaseModel):
    """Demo configuration for multi-turn demonstrations."""

    prompts: list[str] = []
    pause_between_prompts: bool = True


class ThinkingConfig(BaseModel):
    """Configuration for Claude 3.7 thinking capability."""

    type: Literal["enabled", "disabled"] = "enabled"
    budget_tokens: int | None = None

    @field_validator("budget_tokens")
    @classmethod
    def validate_budget(cls, v):
        if v is None:
            return v
        if v < 0:
            raise ValueError("thinking.budget_tokens must be non-negative")
        if 0 < v < 1024:
            warnings.warn(
                f"parameters.thinking.budget_tokens set to {v}, but Claude requires minimum 1024. This will likely fail at runtime.",
                stacklevel=2,
            )
        return v


class ParameterConfig(BaseModel):
    """Validated LLM API parameters."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    top_k: int | None = None
    stop: list[str] | str | None = None
    reasoning_effort: str | None = None
    max_completion_tokens: int | None = None
    max_tokens_to_sample: int | None = None
    stop_sequences: list[str] | str | None = None
    thinking: ThinkingConfig | None = None
    extra_headers: dict[str, Any] | None = None

    model_config = {"extra": "allow"}

    @field_validator("reasoning_effort")
    @classmethod
    def _validate_reasoning_effort(cls, v):
        if v is None:
            return v
        valid_values = {"low", "medium", "high"}
        if v not in valid_values:
            raise ValueError(f"Invalid reasoning_effort value '{v}'. Must be one of: {', '.join(valid_values)}")
        return v

    @model_validator(mode="after")
    def _validate_conflicts(self):
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError(
                "Cannot specify both 'max_tokens' and 'max_completion_tokens'. Use 'max_tokens' for standard models and 'max_completion_tokens' for reasoning models."
            )

        if self.model_extra:
            for key in self.model_extra:
                warnings.warn(
                    f"Unknown API parameter '{key}' in configuration. This may be a typo or a newer parameter not yet recognized.",
                    stacklevel=2,
                )
        return self


class LLMProgramConfig(BaseModel):
    """Full LLM program configuration."""

    model: ModelConfig
    prompt: PromptConfig = PromptConfig()
    parameters: ParameterConfig = Field(default_factory=ParameterConfig)
    mcp: MCPConfig | None = None
    tools: ToolsConfig | None = ToolsConfig()
    plugins: PluginsConfig | None = None
    demo: DemoConfig | None = None

    model_config = {"extra": "allow"}

    # Legacy file descriptor validation removed - FD tools now come from plugins

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
        return self.parameters.model_dump(exclude_none=True)
