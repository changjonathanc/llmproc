"""Configuration schema for LLM programs using Pydantic models."""

from typing import Any

from pydantic import (
    BaseModel,
    RootModel,
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
        from llmproc.config.utils import resolve_path

        # First check for system_prompt_file (takes precedence)
        if self.system_prompt_file:
            try:
                file_path = resolve_path(
                    self.system_prompt_file,
                    base_dir,
                    must_exist=True,
                    error_prefix="System prompt file"
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
    linked_programs: LinkedProgramsConfig | None = LinkedProgramsConfig()
    file_descriptor: FileDescriptorConfig | None = None

    model_config = {
        "extra": "forbid"  # Forbid extra fields
    }
    
    @field_validator("parameters")
    def validate_reasoning_parameters(cls, v):
        """Validate that reasoning parameters have valid values."""
        # Check reasoning_effort values
        if "reasoning_effort" in v:
            valid_values = {"low", "medium", "high"}
            if v["reasoning_effort"] not in valid_values:
                raise ValueError(
                    f"Invalid reasoning_effort value '{v['reasoning_effort']}'. "
                    f"Must be one of: {', '.join(valid_values)}"
                )
        
        # Validate thinking_budget for Claude 3.7
        if "thinking_budget" in v:
            thinking_budget = v["thinking_budget"]
            # 0 is allowed to explicitly disable thinking
            if thinking_budget < 0:
                raise ValueError(
                    f"Invalid thinking_budget value '{thinking_budget}'. "
                    f"Must be 0 (to disable) or at least 1024."
                )
            # Warn if thinking_budget is between 1-1023 as Claude requires minimum 1024
            elif 0 < thinking_budget < 1024:
                import warnings
                warnings.warn(
                    f"thinking_budget set to {thinking_budget}, but Claude requires minimum 1024. "
                    f"This will likely fail at runtime.",
                    stacklevel=2
                )
        
        # Check for token parameter conflicts
        if "max_tokens" in v and "max_completion_tokens" in v:
            raise ValueError(
                "Cannot specify both 'max_tokens' and 'max_completion_tokens'. "
                "Use 'max_tokens' for standard models and 'max_completion_tokens' for reasoning models."
            )
        
        return v
    
    @model_validator(mode="after")
    def validate_file_descriptor(self):
        """Validate file descriptor configuration is consistent with tools.
        
        This validator checks if file_descriptor.enabled is true but no FD tools are enabled.
        It also issues a warning if there's a file_descriptor section but no FD tools.
        """
        import warnings
        
        # FD tools
        fd_tools = ["read_fd", "fd_to_file"]
        
        # Check if file_descriptor is configured
        if self.file_descriptor:
            # Check if any FD tools are enabled
            has_fd_tools = False
            if self.tools and self.tools.enabled:
                has_fd_tools = any(tool in fd_tools for tool in self.tools.enabled)
            
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
                    stacklevel=2
                )
        
        return self

    @model_validator(mode="after")
    def validate_parameters(self):
        """Validate the parameters dictionary and issue warnings for unknown parameters.

        This validator doesn't reject unknown parameters, it just issues warnings.
        We want to stay flexible as LLM APIs evolve, but provide guidance on what's expected.
        """
        import warnings

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
            "thinking_budget",  # For Claude 3.7+ thinking models
        }

        if self.parameters:
            for param_name in self.parameters:
                if param_name not in known_parameters:
                    warnings.warn(
                        f"Unknown API parameter '{param_name}' in configuration. "
                        f"This may be a typo or a newer parameter not yet recognized.",
                        stacklevel=2
                    )
            
            # Check if using OpenAI reasoning model - in this case we need special parameter handling
            is_reasoning_model = False
            if hasattr(self, "model") and self.model.provider == "openai":
                is_reasoning_model = self.model.name.startswith(("o1", "o3"))
                
                # If using a reasoning model, suggest recommended parameters
                if is_reasoning_model and "reasoning_effort" not in self.parameters:
                    warnings.warn(
                        "OpenAI reasoning model detected (o1, o3). For better results, "
                        "consider adding the 'reasoning_effort' parameter (low, medium, high).",
                        stacklevel=2
                    )
            
            # Check if using Claude 3.7 thinking model
            is_claude_thinking_model = False
            if hasattr(self, "model") and self.model.provider == "anthropic":
                is_claude_thinking_model = self.model.name.startswith("claude-3-7")
                
                # If using a Claude thinking model, suggest recommended parameters
                if is_claude_thinking_model and "thinking_budget" not in self.parameters:
                    warnings.warn(
                        "Claude 3.7+ thinking model detected. For better results, "
                        "consider adding the 'thinking_budget' parameter (minimum 1024).",
                        stacklevel=2
                    )
                
            # Check if reasoning_effort used with non-OpenAI provider
            if "reasoning_effort" in self.parameters and hasattr(self, "model") and self.model.provider != "openai":
                warnings.warn(
                    "The 'reasoning_effort' parameter is only supported with OpenAI reasoning models. "
                    "It will be ignored for other providers.",
                    stacklevel=2
                )
                
            # Check if thinking_budget used with non-Claude provider or non-3.7 Claude
            if "thinking_budget" in self.parameters and hasattr(self, "model") and \
               (self.model.provider != "anthropic" or not self.model.name.startswith("claude-3-7")):
                warnings.warn(
                    "The 'thinking_budget' parameter is only supported with Claude 3.7+ models. "
                    "It will be ignored for other providers.",
                    stacklevel=2
                )
                
            # Validate that reasoning models use max_completion_tokens
            if hasattr(self, "model") and self.model.provider == "openai":
                if is_reasoning_model and "max_tokens" in self.parameters:
                    warnings.warn(
                        "OpenAI reasoning models (o1, o3) should use 'max_completion_tokens' instead of 'max_tokens'. "
                        "Your configuration may fail at runtime.",
                        stacklevel=2
                    )
                elif not is_reasoning_model and "max_completion_tokens" in self.parameters:
                    warnings.warn(
                        "'max_completion_tokens' is only for OpenAI reasoning models (o1, o3). "
                        "Standard models should use 'max_tokens' instead.",
                        stacklevel=2
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
