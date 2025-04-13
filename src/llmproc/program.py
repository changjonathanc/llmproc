"""LLMProgram compiler for validating and loading LLM program configurations."""

import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, Union

import llmproc
from llmproc._program_docs import (
    ADD_LINKED_PROGRAM,
    ADD_PRELOAD_FILE,
    ADD_TOOL,
    API_PARAMS,
    COMPILE,
    COMPILE_SELF,
    CONFIGURE_ENV_INFO,
    CONFIGURE_FILE_DESCRIPTOR,
    CONFIGURE_MCP,
    CONFIGURE_THINKING,
    ENABLE_TOKEN_EFFICIENT_TOOLS,
    FROM_TOML,
    INIT,
    LLMPROGRAM_CLASS,
    SET_ENABLED_TOOLS,
    SET_TOOL_ALIASES,
)
from llmproc.env_info import EnvInfoBuilder

# Set up logger
logger = logging.getLogger(__name__)


# Global singleton registry for compiled programs
class ProgramRegistry:
    """Global registry for compiled programs to avoid duplicate compilation."""

    _instance = None

    def __new__(cls):
        """Create a singleton instance of ProgramRegistry."""
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
    """Program definition for LLM processes."""

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
        project_id: str | None = None,
        region: str | None = None,
    ):
        """Initialize a program."""
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
        self.project_id = project_id
        self.region = region
        self.parameters = parameters or {}
        self.display_name = display_name or f"{provider.title()} {model_name}"
        self.preload_files = preload_files or []
        self.mcp_config_path = mcp_config_path
        self.disable_automatic_caching = disable_automatic_caching
        self.mcp_tools = mcp_tools or {}

        # Initialize the tool manager
        from llmproc.tools import ToolManager

        self.tool_manager = ToolManager()

        # Handle tools which can be a dict or a list of function-based tools
        self.tools = {}
        if tools:
            if isinstance(tools, dict):
                # Create a copy of the tools dict to avoid modifying the input
                self.tools = tools.copy()

                # Process enabled tools if specified
                if "enabled" in tools and isinstance(tools["enabled"], list):
                    # Call our own set_enabled_tools which delegates to ToolManager
                    self.set_enabled_tools(tools["enabled"])

                # Register aliases if specified
                if "aliases" in tools and isinstance(tools["aliases"], dict):
                    self.tool_manager.register_aliases(tools["aliases"])
            else:
                # Default to empty tools dictionary if invalid format provided
                import warnings

                warnings.warn(
                    f"Invalid format for tools parameter: expected dict, got {type(tools)}. "
                    + "Defaulting to empty tools configuration.",
                    UserWarning,
                    stacklevel=2,
                )
                self.tools = {"enabled": [], "aliases": {}}

        self.linked_programs = linked_programs or {}
        self.linked_program_descriptions = linked_program_descriptions or {}
        self.env_info = env_info or {
            "variables": []
        }  # Default to empty list (disabled)
        self.file_descriptor = file_descriptor or {}
        self.base_dir = base_dir

    def _compile_self(self) -> "LLMProgram":
        """Internal method to validate and compile this program."""
        # Skip if already compiled
        if self.compiled:
            return self

        # Resolve system prompt from file if specified
        if self._system_prompt_file and not self.system_prompt:
            try:
                with open(self._system_prompt_file) as f:
                    self.system_prompt = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"System prompt file not found: {self._system_prompt_file}"
                )

        # Validate required fields
        if not self.model_name or not self.provider or not self.system_prompt:
            missing = []
            if not self.model_name:
                missing.append("model_name")
            if not self.provider:
                missing.append("provider")
            if not self.system_prompt:
                missing.append("system_prompt or system_prompt_file")
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        # Tool management is now handled directly by the ToolManager
        # Process function tools to ensure they're properly prepared for registration
        self.tool_manager.process_function_tools()

        # Resolve File Descriptor and Tools dependencies
        self._resolve_fd_tool_dependencies()

        # Handle linked programs recursively
        self._compile_linked_programs()

        # Mark as compiled
        self.compiled = True
        return self

    def _resolve_fd_tool_dependencies(self) -> None:
        """Resolve dependencies between file descriptor system and FD tools.

        This method ensures consistency between:
        1. File descriptor system configuration (self.file_descriptor)
        2. Enabled tools that interact with file descriptors (read_fd, fd_to_file)

        The rules are:
        - If FD system is enabled, ensure read_fd tool is available
        - If FD tools are enabled but FD system isn't, enable the FD system
        """
        from llmproc.file_descriptors.constants import FD_RELATED_TOOLS

        # Get current state
        has_fd_config = hasattr(self, "file_descriptor") and isinstance(
            self.file_descriptor, dict
        )
        fd_enabled = has_fd_config and self.file_descriptor.get("enabled", False)
        enabled_tools = self.tool_manager.get_enabled_tools()
        has_fd_tools = any(tool in FD_RELATED_TOOLS for tool in enabled_tools)

        if fd_enabled and not has_fd_tools:
            # If FD system is enabled but no FD tools, add read_fd
            if "read_fd" not in enabled_tools:
                new_enabled_tools = enabled_tools + ["read_fd"]
                self.tool_manager.set_enabled_tools(new_enabled_tools)
                # No need to update self.tools["enabled"] as tool_manager is the source of truth
                logger.info(
                    "File descriptor system enabled, automatically adding read_fd tool"
                )

        elif has_fd_tools and not fd_enabled:
            # If FD tools are enabled but FD system isn't, enable the FD system
            if not has_fd_config:
                self.file_descriptor = {"enabled": True}
            else:
                self.file_descriptor["enabled"] = True
            logger.info(
                "FD tools enabled, automatically enabling file descriptor system"
            )

    def _compile_linked_programs(self) -> None:
        """Compile linked programs recursively."""
        compiled_linked = {}

        # Process each linked program
        for name, program_or_path in self.linked_programs.items():
            if isinstance(program_or_path, str):
                # It's a path, load and compile using from_toml
                try:
                    linked_program = LLMProgram.from_toml(program_or_path)
                    compiled_linked[name] = linked_program
                except FileNotFoundError:
                    warnings.warn(
                        f"Linked program not found: {program_or_path}", stacklevel=2
                    )
            elif isinstance(program_or_path, LLMProgram):
                # It's already a program instance, compile it if not already compiled
                if not program_or_path.compiled:
                    program_or_path._compile_self()
                compiled_linked[name] = program_or_path
            else:
                raise ValueError(
                    f"Invalid linked program type for {name}: {type(program_or_path)}"
                )

        # Replace linked_programs with compiled versions
        self.linked_programs = compiled_linked

    def add_linked_program(
        self, name: str, program: "LLMProgram", description: str = ""
    ) -> "LLMProgram":
        """Link another program to this one."""
        self.linked_programs[name] = program
        self.linked_program_descriptions[name] = description
        return self

    def add_preload_file(self, file_path: str) -> "LLMProgram":
        """Add a file to preload into the system prompt."""
        self.preload_files.append(file_path)
        return self

    def configure_env_info(self, variables: list[str] | str = "all") -> "LLMProgram":
        """Configure environment information sharing."""
        if variables == "all":
            self.env_info = {"variables": "all"}
        else:
            self.env_info = {"variables": variables}
        return self

    def configure_file_descriptor(
        self,
        enabled: bool = True,
        max_direct_output_chars: int = 8000,
        default_page_size: int = 4000,
        max_input_chars: int = 8000,
        page_user_input: bool = True,
        enable_references: bool = True,
    ) -> "LLMProgram":
        """Configure the file descriptor system."""
        self.file_descriptor = {
            "enabled": enabled,
            "max_direct_output_chars": max_direct_output_chars,
            "default_page_size": default_page_size,
            "max_input_chars": max_input_chars,
            "page_user_input": page_user_input,
            "enable_references": enable_references,
        }
        return self

    def configure_thinking(
        self, enabled: bool = True, budget_tokens: int = 4096
    ) -> "LLMProgram":
        """Configure Claude 3.7 thinking capability."""
        # Ensure parameters dict exists
        if self.parameters is None:
            self.parameters = {}

        # Configure thinking
        self.parameters["thinking"] = {
            "type": "enabled" if enabled else "disabled",
            "budget_tokens": budget_tokens,
        }
        return self

    def enable_token_efficient_tools(self) -> "LLMProgram":
        """Enable token-efficient tool use for Claude 3.7 models."""
        # Ensure parameters dict exists
        if self.parameters is None:
            self.parameters = {}

        # Ensure extra_headers dict exists
        if "extra_headers" not in self.parameters:
            self.parameters["extra_headers"] = {}

        # Add header for token-efficient tools
        self.parameters["extra_headers"]["anthropic-beta"] = (
            "token-efficient-tools-2025-02-19"
        )
        return self

    def set_enabled_tools(self, tools: list[Union[str, Callable]]) -> "LLMProgram":
        """Sets the list of enabled tools, replacing any previous list.

        Accepts tool names (str) or callable functions. Callables will be
        added to the ToolManager's function list if not already present.

        Args:
            tools: A list of tool names (str) or functions (Callable) to enable.

        Returns:
            self (for method chaining)
        """
        if not isinstance(tools, list):
            raise ValueError(
                f"Expected a list of tools (strings or callables), got {type(tools)}"
            )

        # Delegate entirely to ToolManager's unified method that handles mixed lists
        # ToolManager is the single source of truth for enabled tools
        self.tool_manager.set_enabled_tools(tools)

        return self

    def get_enabled_tools(self) -> list[str]:
        """Get the list of enabled tool names.

        Returns:
            A list of the currently enabled tool names

        Note:
            This method delegates to the tool_manager, which is the
            single source of truth for enabled tools.
        """
        return self.tool_manager.get_enabled_tools()

    def set_tool_aliases(self, aliases: dict[str, str]) -> "LLMProgram":
        """Set tool aliases, merging with any existing aliases."""
        # Validate aliases is a dictionary
        if not isinstance(aliases, dict):
            raise ValueError(f"Expected dictionary of aliases, got {type(aliases)}")

        # Check for one-to-one mapping (no multiple aliases to same target)
        targets = {}
        for alias, target in aliases.items():
            if target in targets:
                raise ValueError(
                    f"Multiple aliases point to the same target tool '{target}': '{targets[target]}' and '{alias}'. One-to-one mapping is required."
                )
            targets[target] = alias

        # Initialize tools dict if needed
        if not isinstance(self.tools, dict):
            self.tools = {}

        if "aliases" not in self.tools:
            self.tools["aliases"] = {}

        # Merge with existing aliases
        self.tools["aliases"].update(aliases)

        return self

    def configure_mcp(
        self, config_path: str, tools: dict[str, list[str] | str] = None
    ) -> "LLMProgram":
        """Configure Model Context Protocol (MCP) tools."""
        self.mcp_config_path = config_path
        if tools:
            self.mcp_tools = tools
        return self

    def compile(self) -> "LLMProgram":
        """Validate and compile this program."""
        # Call the internal _compile_self method
        return self._compile_self()

    @property
    def api_params(self) -> dict[str, Any]:
        """Get API parameters for LLM API calls."""
        return self.parameters.copy() if self.parameters else {}

    @classmethod
    def from_toml(cls, toml_file, **kwargs):
        """Create a program from a TOML file.

        This method delegates to ProgramLoader.from_toml for backward compatibility.

        Args:
            toml_file: Path to the TOML file
            **kwargs: Additional parameters to override TOML values

        Returns:
            An initialized LLMProgram instance
        """
        from llmproc.config.program_loader import ProgramLoader

        return ProgramLoader.from_toml(toml_file, **kwargs)

    def get_tool_configuration(
        self, linked_programs_instances: dict[str, Any] | None = None
    ) -> dict:
        """Create tool configuration dictionary for initialization.

        This method extracts the necessary components from the program to initialize
        tools without requiring a process instance, avoiding circular dependencies.

        Args:
            linked_programs_instances: Dictionary of pre-initialized LLMProcess instances

        Returns:
            Dictionary with tool configuration components
        """
        # Ensure the program is compiled
        if not self.compiled:
            self.compile()

        # Extract core configuration properties
        config = {
            "provider": self.provider,
            "mcp_config_path": getattr(self, "mcp_config_path", None),
            "mcp_tools": getattr(self, "mcp_tools", {}),
            "mcp_enabled": getattr(self, "mcp_config_path", None) is not None,
        }

        # Handle linked programs
        linked_programs = {}
        if linked_programs_instances:
            linked_programs = linked_programs_instances
            config["has_linked_programs"] = bool(linked_programs)
        elif hasattr(self, "linked_programs") and self.linked_programs:
            linked_programs = self.linked_programs
            config["has_linked_programs"] = True
        else:
            config["has_linked_programs"] = False

        config["linked_programs"] = linked_programs

        # Add linked program descriptions if available
        if (
            hasattr(self, "linked_program_descriptions")
            and self.linked_program_descriptions
        ):
            config["linked_program_descriptions"] = self.linked_program_descriptions
        else:
            config["linked_program_descriptions"] = {}

        # Create file descriptor manager if needed
        fd_manager = None
        if hasattr(self, "file_descriptor"):
            fd_config = self.file_descriptor
            enabled = fd_config.get("enabled", False)

            if enabled:
                # Get configuration values with defaults
                default_page_size = fd_config.get("default_page_size", 4000)
                max_direct_output_chars = fd_config.get("max_direct_output_chars", 8000)
                max_input_chars = fd_config.get("max_input_chars", 8000)
                page_user_input = fd_config.get("page_user_input", True)
                enable_references = fd_config.get("enable_references", False)

                # Create fd_manager
                from llmproc.file_descriptors.manager import FileDescriptorManager

                fd_manager = FileDescriptorManager(
                    default_page_size=default_page_size,
                    max_direct_output_chars=max_direct_output_chars,
                    max_input_chars=max_input_chars,
                    page_user_input=page_user_input,
                    enable_references=enable_references,
                )

                config["references_enabled"] = enable_references

        config["fd_manager"] = fd_manager
        config["file_descriptor_enabled"] = fd_manager is not None

        logger.info("Created tool configuration for initialization")
        return config

    async def start(self) -> "LLMProcess":  # noqa: F821
        """Create and fully initialize an LLMProcess from this program.

        ✅ THIS IS THE CORRECT WAY TO CREATE AN LLMPROCESS ✅

        ```python
        program = LLMProgram.from_toml("config.toml")
        process = await program.start()  # Correct initialization pattern
        ```

        This method delegates the entire program-to-process creation logic
        to the `llmproc.program_exec.create_process` function, which handles
        compilation, tool initialization, process instantiation, and runtime
        context setup in a modular way.

        ⚠️ IMPORTANT: Never use direct constructor `LLMProcess(program=...)` ⚠️
        Direct instantiation will result in broken context-aware tools (spawn, goto, fd_tools, etc.)
        and bypass the proper tool initialization sequence.

        Returns:
            A fully initialized LLMProcess ready for execution with properly configured tools
        """
        # Delegate to the modular implementation in program_exec.py
        from llmproc.program_exec import create_process

        return await create_process(self)


# Apply full docstrings to class and methods
LLMProgram.__doc__ = LLMPROGRAM_CLASS
LLMProgram.__init__.__doc__ = INIT
LLMProgram._compile_self.__doc__ = COMPILE_SELF
LLMProgram.add_linked_program.__doc__ = ADD_LINKED_PROGRAM
LLMProgram.add_preload_file.__doc__ = ADD_PRELOAD_FILE
LLMProgram.configure_env_info.__doc__ = CONFIGURE_ENV_INFO
LLMProgram.configure_file_descriptor.__doc__ = CONFIGURE_FILE_DESCRIPTOR
LLMProgram.configure_thinking.__doc__ = CONFIGURE_THINKING
LLMProgram.enable_token_efficient_tools.__doc__ = ENABLE_TOKEN_EFFICIENT_TOOLS
LLMProgram.set_enabled_tools.__doc__ = SET_ENABLED_TOOLS
LLMProgram.set_tool_aliases.__doc__ = SET_TOOL_ALIASES
LLMProgram.configure_mcp.__doc__ = CONFIGURE_MCP
LLMProgram.compile.__doc__ = COMPILE
LLMProgram.api_params.__doc__ = API_PARAMS
# Skip from_toml as it's a staticmethod and docs can't be assigned
