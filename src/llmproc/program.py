"""LLMProgram compiler for validating and loading LLM program configurations."""

import tomllib
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import ValidationError

from llmproc.config.schema import LLMProgramConfig
from llmproc.config.utils import resolve_path
from llmproc.env_info import EnvInfoBuilder


# Global singleton registry for compiled programs
class ProgramRegistry:
    """Global registry for compiled programs to avoid duplicate compilation."""

    _instance = None

    def __new__(cls):
        """Create a singleton instance of ProgramRegistry.

        Returns:
            The singleton ProgramRegistry instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._compiled_programs = {}
        return cls._instance

    def register(self, path: Path, program: "LLMProgram") -> None:
        """Register a compiled program."""
        self._compiled_programs[str(path.resolve())] = program

    def get(self, path: Path) -> Optional["LLMProgram"]:
        """Get a compiled program if it exists."""
        return self._compiled_programs.get(str(path.resolve()))

    def contains(self, path: Path) -> bool:
        """Check if a program has been compiled."""
        return str(path.resolve()) in self._compiled_programs

    def clear(self) -> None:
        """Clear all compiled programs (mainly for testing)."""
        self._compiled_programs.clear()


class LLMProgram:
    """Program definition for LLM processes.

    This class handles creating, configuring, and compiling LLM programs
    for use with LLMProcess. It supports both direct initialization in code
    and loading from TOML configuration files.
    """

    def __init__(
        self,
        model_name: str,
        provider: str,
        system_prompt: str = None,
        system_prompt_file: str = None,
        parameters: dict[str, Any] = None,
        display_name: str | None = None,
        preload_files: list[str] | None = None,
        mcp_config_path: str | None = None,
        mcp_tools: dict[str, list[str]] | None = None,
        tools: dict[str, Any] | list[Any] | None = None,
        linked_programs: dict[str, Union[str, "LLMProgram"]] | None = None,
        linked_program_descriptions: dict[str, str] | None = None,
        env_info: dict[str, Any] | None = None,
        file_descriptor: dict[str, Any] | None = None,
        base_dir: Path | None = None,
        disable_automatic_caching: bool = False,
    ):
        """Initialize a program.

        Args:
            model_name: Name of the model to use
            provider: Provider of the model (openai, anthropic, or anthropic_vertex)
            system_prompt: System prompt text that defines the behavior of the process
            system_prompt_file: Path to a file containing the system prompt (alternative to system_prompt)
            parameters: Dictionary of API parameters
            display_name: User-facing name for the process in CLI interfaces
            preload_files: List of file paths to preload into the system prompt as context
            mcp_config_path: Path to MCP servers configuration file
            mcp_tools: Dictionary mapping server names to tools to enable
            tools: Dictionary from the [tools] section, or list of function-based tools
            linked_programs: Dictionary mapping program names to paths or LLMProgram objects
            linked_program_descriptions: Dictionary mapping program names to descriptions
            env_info: Environment information configuration
            file_descriptor: File descriptor configuration
            base_dir: Base directory for resolving relative paths in files
            disable_automatic_caching: Whether to disable automatic prompt caching for Anthropic models
        """
        # Flag to track if this program has been fully compiled
        self.compiled = False
        self._system_prompt_file = system_prompt_file
        
        # Handle system prompt (either direct or from file)
        if system_prompt and system_prompt_file:
            raise ValueError("Cannot specify both system_prompt and system_prompt_file")
            
        # Initialize core attributes
        self.model_name = model_name
        self.provider = provider
        self.system_prompt = system_prompt
        self.parameters = parameters or {}
        self.display_name = display_name or f"{provider.title()} {model_name}"
        self.preload_files = preload_files or []
        self.mcp_config_path = mcp_config_path
        self.disable_automatic_caching = disable_automatic_caching
        self.mcp_tools = mcp_tools or {}
        
        # Handle tools which can be a dict or a list of function-based tools
        self.tools = {}
        if tools:
            if isinstance(tools, dict):
                self.tools = tools
            elif isinstance(tools, list):
                # Will handle function-based tools in a future implementation
                # For now, just store them as a list
                self._function_tools = tools
                # Enable tools section with empty enabled list to be populated later
                self.tools = {"enabled": []}
        
        self.linked_programs = linked_programs or {}
        self.linked_program_descriptions = linked_program_descriptions or {}
        self.env_info = env_info or {
            "variables": []
        }  # Default to empty list (disabled)
        self.file_descriptor = file_descriptor or {}
        self.base_dir = base_dir

    def _compile_self(self) -> "LLMProgram":
        """Internal method to validate and compile this program.

        This method validates the program configuration, resolves any
        system prompt files, and compiles linked programs recursively.
        
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If validation fails
            FileNotFoundError: If required files cannot be found
        """
        # Skip if already compiled
        if self.compiled:
            return self
            
        # Resolve system prompt from file if specified
        if self._system_prompt_file and not self.system_prompt:
            try:
                with open(self._system_prompt_file, 'r') as f:
                    self.system_prompt = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(f"System prompt file not found: {self._system_prompt_file}")
                
        # Validate required fields
        if not self.model_name:
            raise ValueError("model_name is required")
        if not self.provider:
            raise ValueError("provider is required")
        if not self.system_prompt:
            raise ValueError("Either system_prompt or system_prompt_file must be provided")
            
        # Handle linked programs
        compiled_linked = {}
        for name, program_or_path in self.linked_programs.items():
            if isinstance(program_or_path, str):
                # It's a path, load and compile using from_toml
                try:
                    linked_program = LLMProgram.from_toml(program_or_path)
                except FileNotFoundError:
                    # Issue a warning but don't fail
                    warnings.warn(f"Linked program not found: {program_or_path}", stacklevel=2)
                    continue
                compiled_linked[name] = linked_program
            elif isinstance(program_or_path, LLMProgram):
                # It's already a program instance, compile it if not already compiled
                if not program_or_path.compiled:
                    program_or_path._compile_self()
                compiled_linked[name] = program_or_path
            else:
                raise ValueError(f"Invalid linked program type for {name}: {type(program_or_path)}")
                
        # Replace linked_programs with compiled versions
        self.linked_programs = compiled_linked
        
        # Mark as compiled
        self.compiled = True
        return self
        
    def link_program(self, name: str, program: "LLMProgram", description: str = "") -> "LLMProgram":
        """Link another program to this one.
        
        Args:
            name: Name to identify the linked program
            program: LLMProgram instance to link
            description: Optional description of the program's purpose
            
        Returns:
            self (for method chaining)
        """
        self.linked_programs[name] = program
        self.linked_program_descriptions[name] = description
        return self
        
    def preload_file(self, file_path: str) -> "LLMProgram":
        """Add a file to preload into the system prompt.
        
        Args:
            file_path: Path to the file to preload
            
        Returns:
            self (for method chaining)
        """
        self.preload_files.append(file_path)
        return self
        
    def add_tool(self, tool) -> "LLMProgram":
        """Add a tool to this program.
        
        Args:
            tool: Either a function to register as a tool, or a tool definition dictionary
            
        Returns:
            self (for method chaining)
        """
        # This will be extended in Phase 2 to handle function-based tools
        if hasattr(self, "_function_tools"):
            self._function_tools.append(tool)
        elif callable(tool):
            # Initialize _function_tools if not present
            self._function_tools = [tool]
        elif isinstance(tool, dict):
            # Add to tools configuration
            if "enabled" not in self.tools:
                self.tools["enabled"] = []
            if "name" in tool and tool["name"] not in self.tools["enabled"]:
                self.tools["enabled"].append(tool["name"])
        
        return self
        
    def compile(self) -> "LLMProgram":
        """Validate and compile this program.
        
        This method validates the program configuration, resolves any
        system prompt files, and compiles linked programs recursively.
        
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If validation fails
            FileNotFoundError: If required files cannot be found
        """
        # Call the internal _compile_self method
        return self._compile_self()
    
    @property
    def api_params(self) -> dict[str, Any]:
        """Get API parameters for LLM API calls.

        This property returns all parameters from the program configuration,
        relying on the schema's validation to issue warnings for unknown parameters.

        Returns:
            Dictionary of API parameters for LLM API calls
        """
        return self.parameters.copy() if self.parameters else {}
        


    @classmethod
    def _compile_single_program(cls, path: Path) -> "LLMProgram":
        """Compile a single program without recursively compiling linked programs.

        Args:
            path: Path to the TOML program file

        Returns:
            Compiled LLMProgram instance
        """
        # Load and parse the TOML file
        with path.open("rb") as f:
            raw_config = tomllib.load(f)

        # Validate the configuration using pydantic
        try:
            config = LLMProgramConfig(**raw_config)
        except ValidationError as e:
            # Enhance the validation error with file information
            error_msg = f"Invalid program configuration in {path}:\n{str(e)}"
            raise ValueError(error_msg)

        # Build the LLMProgram from the validated configuration
        program = cls._build_from_config(config, path.parent)

        # Store the source path
        program.source_path = path

        return program

    @classmethod
    def _build_from_config(
        cls, config: LLMProgramConfig, base_dir: Path
    ) -> "LLMProgram":
        """Build an LLMProgram from a validated configuration.

        Args:
            config: Validated program configuration
            base_dir: Base directory for resolving relative file paths

        Returns:
            Constructed LLMProgram instance

        Raises:
            FileNotFoundError: If required files cannot be found
        """
        # Resolve system prompt using the PromptConfig's resolve method
        # This will raise FileNotFoundError if the system prompt file is specified but not found
        system_prompt = config.prompt.resolve(base_dir)

        # Resolve preload files
        preload_files = None
        if config.preload and config.preload.files:
            preload_files = []
            for file_path in config.preload.files:
                try:
                    # Try to resolve the path but don't require existence
                    resolved_path = resolve_path(file_path, base_dir, must_exist=False)
                    if not resolved_path.exists():
                        # Issue a warning if the file doesn't exist
                        warnings.warn(
                            f"Preload file not found - Specified: '{file_path}', Resolved: '{resolved_path}'",
                            stacklevel=2
                        )
                    # Include the file path regardless - it will be checked at runtime when actually loaded
                    preload_files.append(str(resolved_path))
                except Exception as e:
                    # If there's any other error in path resolution, issue a warning
                    warnings.warn(
                        f"Error resolving preload file path '{file_path}': {str(e)}",
                        stacklevel=2
                    )

        # Resolve MCP configuration
        mcp_config_path = None
        if config.mcp and config.mcp.config_path:
            try:
                mcp_path = resolve_path(
                    config.mcp.config_path,
                    base_dir,
                    must_exist=True,
                    error_prefix="MCP config file"
                )
                mcp_config_path = str(mcp_path)
            except FileNotFoundError as e:
                # Re-raise with the original error message
                raise FileNotFoundError(str(e))

        # Extract MCP tools configuration
        mcp_tools = None
        if config.mcp and config.mcp.tools:
            mcp_tools = config.mcp.tools.root

        # Process linked programs
        linked_programs = None
        linked_program_descriptions = None
        if config.linked_programs:
            linked_programs = {}
            linked_program_descriptions = {}
            for program_name, program_config in config.linked_programs.root.items():
                # Handle both string paths and LinkedProgramItem objects
                if isinstance(program_config, str):
                    linked_programs[program_name] = program_config
                    linked_program_descriptions[program_name] = ""
                else:
                    # It's a LinkedProgramItem
                    linked_programs[program_name] = program_config.path
                    linked_program_descriptions[program_name] = program_config.description

        # Extract environment info configuration
        env_info = (
            config.env_info.model_dump() if config.env_info else {"variables": []}
        )
        
        # Extract file descriptor configuration
        file_descriptor = (
            config.file_descriptor.model_dump() if config.file_descriptor else None
        )

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
            linked_program_descriptions=linked_program_descriptions,
            env_info=env_info,
            file_descriptor=file_descriptor,
            base_dir=base_dir,
            disable_automatic_caching=config.model.disable_automatic_caching,
        )

        return program

    def get_enriched_system_prompt(self, process_instance=None, include_env=True):
        """Get enhanced system prompt with preloaded files and environment info.

        This combines the basic system prompt with preloaded files and optional
        environment information based on the env_info configuration.

        Args:
            process_instance: Optional LLMProcess instance for accessing preloaded content
            include_env: Whether to include environment information (default: True)

        Returns:
            Complete system prompt ready for API calls
        """
        # Use the EnvInfoBuilder to handle environment information and preloaded content
        preloaded_content = {}
        file_descriptor_enabled = False
        references_enabled = False
        page_user_input = False
        
        if process_instance:
            if hasattr(process_instance, "preloaded_content"):
                preloaded_content = process_instance.preloaded_content
            if hasattr(process_instance, "file_descriptor_enabled"):
                file_descriptor_enabled = process_instance.file_descriptor_enabled
            if hasattr(process_instance, "references_enabled"):
                references_enabled = process_instance.references_enabled
                
            # Check if user input paging is enabled
            if hasattr(process_instance, "fd_manager"):
                page_user_input = getattr(process_instance.fd_manager, "page_user_input", False)

        return EnvInfoBuilder.get_enriched_system_prompt(
            base_prompt=self.system_prompt,
            env_config=self.env_info,
            preloaded_content=preloaded_content,
            include_env=include_env,
            file_descriptor_enabled=file_descriptor_enabled,
            references_enabled=references_enabled,
            page_user_input=page_user_input
        )

    @classmethod
    def from_toml(
        cls, toml_path: str | Path, include_linked: bool = True
    ) -> "LLMProgram":
        """Load and compile a program from a TOML file.

        This method loads a TOML configuration file, validates it, and 
        returns a compiled LLMProgram ready to be started.

        Args:
            toml_path: Path to the TOML program file
            include_linked: Whether to compile linked programs recursively

        Returns:
            A compiled LLMProgram ready to start

        Raises:
            FileNotFoundError: If the TOML file doesn't exist
            ValueError: If the configuration is invalid
        """
        # Use the utility function to resolve the path
        path = resolve_path(toml_path, must_exist=True, error_prefix="Program file")

        # Get the global registry
        registry = ProgramRegistry()

        # If already compiled, return from registry
        if registry.contains(path):
            return registry.get(path)

        # Create the uncompiled program
        program = cls._compile_single_program(path)
        registry.register(path, program)

        # If linked programs are requested, handle them now
        if include_linked and program.linked_programs:
            # Process all linked programs that are paths
            for name, program_or_path in list(program.linked_programs.items()):
                if isinstance(program_or_path, str):
                    # It's a path, convert to absolute if needed
                    base_dir = path.parent
                    try:
                        linked_path = resolve_path(
                            program_or_path, 
                            base_dir=base_dir, 
                            must_exist=True,
                            error_prefix=f"Linked program file (from '{path}')"
                        )
                        # Load and compile the linked program
                        linked_program = cls.from_toml(linked_path, include_linked=True)
                        program.linked_programs[name] = linked_program
                    except FileNotFoundError as e:
                        # Re-raise with the original error message
                        raise FileNotFoundError(str(e))

        # Mark the program as compiled
        program.compiled = True
        return program

    async def start(self) -> "LLMProcess":  # noqa: F821
        """Create and fully initialize an LLMProcess from this program.

        This method:
        1. Ensures the program is compiled
        2. Creates an LLMProcess instance
        3. Initializes it asynchronously (for MCP tools, etc.)
        4. Sets up linked programs

        The process will have access to all linked programs that were included
        during compilation. Linked programs are not instantiated until needed.

        Returns:
            A fully initialized LLMProcess ready to run

        Raises:
            RuntimeError: If initialization fails
            ValueError: If program compilation fails
        """
        # Ensure the program is compiled
        if not self.compiled:
            self.compile()
            
        # Import dynamically to avoid circular imports
        import llmproc

        # Create a process and fully initialize it asynchronously
        process = await llmproc.LLMProcess.create(program=self)

        # Ensure linked programs are properly registered if they exist
        if hasattr(self, "linked_programs") and self.linked_programs:
            process.has_linked_programs = True
            
        # Pass along linked program descriptions if they exist
        if hasattr(self, "linked_program_descriptions") and self.linked_program_descriptions:
            process.linked_program_descriptions = self.linked_program_descriptions

        return process
        
    def get_structure(self) -> dict:
        """Return a dictionary representing the structure of this program.
        
        This method is useful for debugging and visualizing the program structure,
        including linked programs and their relationships.
        
        Returns:
            Dictionary with program structure information
        """
        # Basic program info
        info = {
            "model": self.model_name,
            "provider": self.provider,
            "compiled": self.compiled
        }
        
        # Add linked program information if present
        if self.linked_programs:
            linked_info = {}
            for name, program in self.linked_programs.items():
                linked_info[name] = {
                    "model": program.model_name,
                    "provider": program.provider
                }
                # Add description if available
                if name in self.linked_program_descriptions:
                    linked_info[name]["description"] = self.linked_program_descriptions[name]
            
            info["linked_programs"] = linked_info
            
        # Add tools information
        if self.tools and "enabled" in self.tools and self.tools["enabled"]:
            info["tools"] = self.tools["enabled"]
            
        return info
