"""LLMProgram compiler for validating and loading LLM program configurations."""

import os
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
    """Compiler for LLM program configurations.

    This class handles loading, validating, and processing TOML program files
    into a format ready for LLMProcess instantiation.
    """

    def __init__(
        self,
        model_name: str,
        provider: str,
        system_prompt: str,
        parameters: dict[str, Any] = None,
        display_name: str | None = None,
        preload_files: list[str] | None = None,
        mcp_config_path: str | None = None,
        mcp_tools: dict[str, list[str]] | None = None,
        tools: dict[str, Any] | None = None,
        linked_programs: dict[str, Union[str, "LLMProgram"]] | None = None,
        env_info: dict[str, Any] | None = None,
        base_dir: Path | None = None,
    ):
        """Initialize a program.

        Args:
            model_name: Name of the model to use
            provider: Provider of the model (openai, anthropic, or anthropic_vertex)
            system_prompt: System prompt that defines the behavior of the process
            parameters: Dictionary of API parameters
            display_name: User-facing name for the process in CLI interfaces
            preload_files: List of file paths to preload into the system prompt as context
            mcp_config_path: Path to MCP servers configuration file
            mcp_tools: Dictionary mapping server names to tools to enable
            tools: Dictionary from the [tools] section in the TOML program
            linked_programs: Dictionary mapping program names to TOML program paths or compiled LLMProgram objects
            env_info: Environment information configuration
            base_dir: Base directory for resolving relative paths in files
        """
        # Flag to track if this program has been fully compiled (including linked programs)
        self.compiled = False

        # Initialize core attributes
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
        self.env_info = env_info or {
            "variables": []
        }  # Default to empty list (disabled)
        self.base_dir = base_dir

        # No need to initialize api_params here - we'll use a property
        
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
    def compile(
        cls,
        toml_path: str | Path,
        include_linked: bool = True,
        check_linked_files: bool = True,
        return_all: bool = False,
    ) -> Union["LLMProgram", dict[str, "LLMProgram"]]:
        """Compile an LLM program from a TOML file.

        This method loads a TOML file, validates the configuration, resolves file paths,
        and returns a compiled LLMProgram ready for instantiation of an LLMProcess.

        If include_linked=True, it will also compile all linked programs recursively
        and update the linked_programs attribute of each program to reference the
        compiled program objects directly.

        Args:
            toml_path: Path to the TOML program file
            include_linked: Whether to compile linked programs recursively
            check_linked_files: Whether to verify linked program files exist
            return_all: Whether to return all compiled programs as a dictionary

        Returns:
            By default: A single compiled LLMProgram instance for the main program
            If return_all=True: Dictionary mapping absolute paths to compiled programs

        Raises:
            FileNotFoundError: If the TOML file or referenced files cannot be found
            ValidationError: If the configuration is invalid
            ValueError: If there are issues with configuration values
        """
        # Use the utility function to resolve the path
        path = resolve_path(toml_path, must_exist=True, error_prefix="Program file")

        # Get the global registry
        registry = ProgramRegistry()

        # Handle single program compilation (no linked programs)
        if not include_linked:
            # If already compiled, return from registry
            if registry.contains(path):
                return registry.get(path)

            # Otherwise compile and register it
            program = cls._compile_single_program(path)
            program.compiled = True
            registry.register(path, program)
            return program

        # Handle BFS compilation of the program graph
        abs_path = str(path)

        # If main program is already compiled and we're not returning all, just return it
        if registry.contains(path) and not return_all:
            return registry.get(path)

        # Stage 1: Compile all programs without resolving linked programs
        # This builds a complete set of all programs in the graph
        to_compile = deque([path])
        compiled_paths = set()  # Track what we've queued to avoid duplicates
        compiled_paths.add(abs_path)

        while to_compile:
            current_path = to_compile.popleft()

            # Skip if already compiled - just continue to queue its linked programs
            if not registry.contains(current_path):
                # Compile the current program
                program = cls._compile_single_program(current_path)
                registry.register(current_path, program)
            else:
                program = registry.get(current_path)

            # Find linked programs and add them to the compilation queue
            if hasattr(program, "linked_programs") and program.linked_programs:
                base_dir = current_path.parent

                # At this stage, linked_programs contains string paths
                for _linked_name, linked_path_str in list(
                    program.linked_programs.items()
                ):
                    # Skip any non-string items (already processed linked programs)
                    if not isinstance(linked_path_str, str):
                        continue
                    
                    # Only validate existence if check_linked_files is True
                    try:
                        linked_path = resolve_path(
                            linked_path_str, 
                            base_dir, 
                            must_exist=check_linked_files,
                            error_prefix=f"Linked program file (from '{current_path}')"
                        )
                    except FileNotFoundError as e:
                        # Re-raise with the original error message
                        raise FileNotFoundError(str(e))
                        
                    # Only add to queue if we haven't seen it before
                    linked_abs_path = str(linked_path)
                    if linked_abs_path not in compiled_paths:
                        to_compile.append(linked_path)
                        compiled_paths.add(linked_abs_path)

        # Stage 2: Update linked_programs to reference compiled program objects
        for compiled_path in compiled_paths:
            program = registry.get(Path(compiled_path))

            # Update any string paths to refer to compiled program objects
            if hasattr(program, "linked_programs") and program.linked_programs:
                base_dir = Path(compiled_path).parent

                # Create a new dict for the updated references
                updated_links = {}

                for linked_name, linked_path_str in program.linked_programs.items():
                    # Skip if it's already a program object not a string
                    if not isinstance(linked_path_str, str):
                        updated_links[linked_name] = linked_path_str
                        continue

                    # Resolve the path using our utility
                    linked_path = resolve_path(linked_path_str, base_dir, must_exist=False)
                    
                    # Get the compiled program from the registry
                    linked_program = registry.get(linked_path)
                    if linked_program:
                        updated_links[linked_name] = linked_program
                    else:
                        # Should never happen if Stage 1 completed successfully
                        warnings.warn(
                            f"Could not find compiled program for {linked_path}"
                        )
                        updated_links[linked_name] = linked_path_str

                # Replace the linked_programs dict with the updated version
                program.linked_programs = updated_links

            # Mark the program as fully compiled
            program.compiled = True

        # Return either the main program or all compiled programs
        if return_all:
            return {path: registry.get(Path(path)) for path in compiled_paths}
        else:
            return registry.get(path)

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
        if config.linked_programs:
            linked_programs = {}
            for program_name, program_path in config.linked_programs.root.items():
                linked_programs[program_name] = program_path

        # Extract environment info configuration
        env_info = (
            config.env_info.model_dump() if config.env_info else {"variables": []}
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
            env_info=env_info,
            base_dir=base_dir,
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
        if process_instance and hasattr(process_instance, "preloaded_content"):
            preloaded_content = process_instance.preloaded_content

        return EnvInfoBuilder.get_enriched_system_prompt(
            base_prompt=self.system_prompt,
            env_config=self.env_info,
            preloaded_content=preloaded_content,
            include_env=include_env
        )

    @classmethod
    def from_toml(
        cls, toml_path: str | Path, include_linked: bool = True
    ) -> "LLMProgram":
        """Load and compile a program from a TOML file.

        This is a convenience method that simply calls compile() with the
        most common options for end users. It handles loading the TOML file,
        validating it, and creating a compiled program ready to be started.

        Args:
            toml_path: Path to the TOML program file
            include_linked: Whether to compile linked programs recursively

        Returns:
            A compiled LLMProgram ready to start

        Raises:
            FileNotFoundError: If the TOML file doesn't exist
            ValueError: If the configuration is invalid
        """
        return cls.compile(
            toml_path, include_linked=include_linked, check_linked_files=True
        )

    async def start(self) -> "LLMProcess":
        """Create and fully initialize an LLMProcess from this program.

        This is the recommended way to create a process from a program, as it
        properly handles async initialization for features like MCP tools.

        The process will have access to all linked programs that were included
        during compilation. Linked programs are not instantiated until needed.

        Returns:
            A fully initialized LLMProcess ready to run

        Raises:
            RuntimeError: If initialization fails
        """
        # Import dynamically to avoid circular imports
        import llmproc

        # Create a process and fully initialize it asynchronously
        process = await llmproc.LLMProcess.create(program=self)

        # Ensure linked programs are properly registered if they exist
        if hasattr(self, "linked_programs") and self.linked_programs:
            process.has_linked_programs = True

        return process