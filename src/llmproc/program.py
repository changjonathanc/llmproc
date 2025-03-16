"""LLMProgram compiler for validating and loading LLM program configurations."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import tomllib
from pydantic import BaseModel, Field, RootModel, ValidationError, field_validator, model_validator


# Pydantic models for program validation
class ModelConfig(BaseModel):
    """Model configuration section."""
    name: str
    provider: str
    display_name: Optional[str] = None

    @field_validator('provider')
    def validate_provider(cls, v):
        """Validate that the provider is supported."""
        supported_providers = {'openai', 'anthropic', 'vertex'}
        if v not in supported_providers:
            raise ValueError(f"Provider '{v}' not supported. Must be one of: {', '.join(supported_providers)}")
        return v


class PromptConfig(BaseModel):
    """Prompt configuration section."""
    system_prompt: Optional[str] = ""
    system_prompt_file: Optional[str] = None

    @model_validator(mode='after')
    def check_prompt_sources(self):
        """Check that at least one prompt source is provided."""
        if not self.system_prompt and not self.system_prompt_file:
            # Set default empty system prompt
            self.system_prompt = ""
        
        return self


class PreloadConfig(BaseModel):
    """Preload configuration section."""
    files: List[str] = []


class MCPToolsConfig(RootModel):
    """MCP tools configuration."""
    root: Dict[str, Union[List[str], str]] = {}

    @field_validator('root')
    def validate_tools(cls, v):
        """Validate that tool configurations are either lists or 'all'."""
        for server, tools in v.items():
            if not isinstance(tools, list) and tools != "all":
                raise ValueError(f"Tool configuration for server '{server}' must be 'all' or a list of tool names")
        return v


class MCPConfig(BaseModel):
    """MCP configuration section."""
    config_path: Optional[str] = None
    tools: Optional[MCPToolsConfig] = None


class ToolsConfig(BaseModel):
    """Tools configuration section."""
    enabled: List[str] = []


class DebugConfig(BaseModel):
    """Debug configuration section."""
    debug_tools: bool = False


class LinkedProgramsConfig(RootModel):
    """Linked programs configuration section."""
    root: Dict[str, str] = {}


class LLMProgramConfig(BaseModel):
    """Full LLM program configuration."""
    model: ModelConfig
    prompt: PromptConfig = PromptConfig()
    parameters: Dict[str, Any] = {}
    preload: Optional[PreloadConfig] = PreloadConfig()
    mcp: Optional[MCPConfig] = None
    tools: Optional[ToolsConfig] = ToolsConfig()
    debug: Optional[DebugConfig] = DebugConfig()
    linked_programs: Optional[LinkedProgramsConfig] = LinkedProgramsConfig()

    model_config = {
        "extra": "forbid"  # Forbid extra fields
    }


class LLMProgram:
    """Compiler for LLM program configurations.
    
    This class handles loading, validating, and processing TOML program files
    into a format ready for LLMProcess instantiation.
    """

    def __init__(self,
                 model_name: str,
                 provider: str,
                 system_prompt: str,
                 parameters: Dict[str, Any] = None,
                 display_name: Optional[str] = None,
                 preload_files: Optional[List[str]] = None,
                 mcp_config_path: Optional[str] = None,
                 mcp_tools: Optional[Dict[str, List[str]]] = None,
                 tools: Optional[Dict[str, Any]] = None,
                 linked_programs: Optional[Dict[str, str]] = None,
                 debug_tools: bool = False,
                 config_dir: Optional[Path] = None):
        """Initialize a program directly from parameters.
        
        This is a convenience constructor, but most users should use the
        compile() class method instead.
        
        Args:
            model_name: Name of the model to use
            provider: Provider of the model (openai, anthropic, or vertex)
            system_prompt: System prompt that defines the behavior of the process
            parameters: Dictionary of API parameters
            display_name: User-facing name for the process in CLI interfaces
            preload_files: List of file paths to preload into the system prompt as context
            mcp_config_path: Path to MCP servers configuration file
            mcp_tools: Dictionary mapping server names to tools to enable
            tools: Dictionary from the [tools] section in the TOML program
            linked_programs: Dictionary mapping program names to TOML program paths
            debug_tools: Enable detailed debugging output for tool execution
            config_dir: Base directory for resolving relative paths in programs
        """
        self.model_name = model_name
        self.provider = provider
        self.system_prompt = system_prompt
        self.parameters = parameters or {}
        self.display_name = display_name or f"{provider.title()} {model_name}"
        self.preload_files = preload_files or []
        self.mcp_config_path = mcp_config_path
        self.mcp_tools = mcp_tools or {}
        self.tools = tools or {}
        self.linked_programs = linked_programs or {}
        self.debug_tools = debug_tools
        self.config_dir = config_dir
        
        # Extract API parameters from parameters for convenience
        self.api_params = self._extract_api_parameters()
        
    def _extract_api_parameters(self) -> Dict[str, Any]:
        """Extract known API parameters from the parameters dictionary.
        
        Returns:
            Dictionary of API parameters for LLM API calls
        """
        api_params = {}
        api_param_keys = [
            "temperature", "max_tokens", "top_p", 
            "frequency_penalty", "presence_penalty"
        ]
        
        for key in api_param_keys:
            if key in self.parameters:
                api_params[key] = self.parameters[key]
                
        return api_params
    
    @classmethod
    def compile(cls, toml_path: Union[str, Path]) -> 'LLMProgram':
        """Compile an LLM program from a TOML file.
        
        This method loads a TOML file, validates the configuration, resolves
        file paths, and returns a compiled LLMProgram ready for instantiation
        of an LLMProcess.
        
        Args:
            toml_path: Path to the TOML program file
            
        Returns:
            Compiled LLMProgram instance
            
        Raises:
            FileNotFoundError: If the TOML file cannot be found
            ValidationError: If the configuration is invalid
            ValueError: If there are issues with file paths or configuration values
        """
        path = Path(toml_path)
        if not path.exists():
            raise FileNotFoundError(f"Program file not found: {toml_path}")
        
        # Load and parse the TOML file
        with path.open("rb") as f:
            raw_config = tomllib.load(f)
        
        # Validate the configuration using pydantic
        try:
            config = LLMProgramConfig(**raw_config)
        except ValidationError as e:
            # Enhance the validation error with file information
            # Just wrap the error with additional context
            error_msg = f"Invalid program configuration in {toml_path}:\n{str(e)}"
            raise ValueError(error_msg)
        
        # Build the LLMProgram from the validated configuration
        program = cls._build_from_config(config, path.parent)
        
        # Store the source path
        program.source_path = path
        
        return program
    
    @classmethod
    def _build_from_config(cls, config: LLMProgramConfig, config_dir: Path) -> 'LLMProgram':
        """Build an LLMProgram from a validated configuration.
        
        Args:
            config: Validated program configuration
            config_dir: Directory containing the program file for resolving paths
            
        Returns:
            Constructed LLMProgram instance
        """
        # Resolve system prompt
        system_prompt = config.prompt.system_prompt
        if config.prompt.system_prompt_file:
            system_prompt_path = config_dir / config.prompt.system_prompt_file
            if not system_prompt_path.exists():
                print(f"<warning>System prompt file {system_prompt_path} does not exist.</warning>")
            else:
                system_prompt = system_prompt_path.read_text()
        
        # Resolve preload files
        preload_files = None
        if config.preload and config.preload.files:
            preload_files = [str(config_dir / file_path) for file_path in config.preload.files]
        
        # Resolve MCP configuration
        mcp_config_path = None
        if config.mcp and config.mcp.config_path:
            mcp_path = config_dir / config.mcp.config_path
            if not mcp_path.exists():
                print(f"<warning>MCP config file {mcp_path} does not exist.</warning>")
            else:
                mcp_config_path = str(mcp_path)
        
        # Extract MCP tools configuration
        mcp_tools = None
        if config.mcp and config.mcp.tools:
            mcp_tools = config.mcp.tools.root
        
        # Process linked programs
        linked_programs = None
        if config.linked_programs:
            linked_programs = {}
            for program_name, program_path in config.linked_programs.root.items():
                linked_programs[program_name] = program_path
        
        # Get debug settings
        debug_tools = config.debug.debug_tools if config.debug else False
        
        # Create the program instance
        program = cls(
            model_name=config.model.name,
            provider=config.model.provider,
            system_prompt=system_prompt,
            parameters=config.parameters,
            display_name=config.model.display_name,
            preload_files=preload_files,
            mcp_config_path=mcp_config_path,
            mcp_tools=mcp_tools,
            tools=config.tools.model_dump() if config.tools else None,
            linked_programs=linked_programs,
            debug_tools=debug_tools,
            config_dir=config_dir
        )
        
        return program
    
    def instantiate(self, from_llmproc):
        """Instantiate an LLMProcess from this program.
        
        Args:
            from_llmproc: The llmproc module to import LLMProcess from
            
        Returns:
            An initialized LLMProcess instance
        """
        # Import dynamically to avoid circular imports
        LLMProcess = from_llmproc.LLMProcess
        
        # Create the LLMProcess instance
        return LLMProcess(
            model_name=self.model_name,
            provider=self.provider,
            system_prompt=self.system_prompt,
            preload_files=self.preload_files,
            display_name=self.display_name,
            mcp_config_path=self.mcp_config_path,
            mcp_tools=self.mcp_tools,
            linked_programs=self.linked_programs,
            config_dir=self.config_dir,
            parameters=self.parameters,
            tools=self.tools,
            debug_tools=self.debug_tools
        )