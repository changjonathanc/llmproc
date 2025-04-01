"""LLMProgram compiler for validating and loading LLM program configurations."""

import logging
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import llmproc
from llmproc.env_info import EnvInfoBuilder

# Set up logger
logger = logging.getLogger(__name__)


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
    for use with LLMProcess. It focuses on the Python SDK interface and
    core functionality, with configuration loading delegated to specialized loaders.
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
        
        # Initialize the tool manager
        from llmproc.tools import ToolManager
        self.tool_manager = ToolManager()
        
        # Handle tools which can be a dict or a list of function-based tools
        self.tools = {}
        if tools:
            if isinstance(tools, dict):
                self.tools = tools
                
                # Add to tool manager if there are enabled tools
                if "enabled" in tools and isinstance(tools["enabled"], list):
                    # Set enabled tools in the tool manager
                    self.tool_manager.set_enabled_tools(tools["enabled"])
                        
            elif isinstance(tools, list):
                # For backward compatibility, store as _function_tools
                self._function_tools = tools
                # Enable tools section with empty enabled list to be populated later
                self.tools = {"enabled": []}
                
                # Add to tool manager
                for func_tool in tools:
                    self.tool_manager.add_function_tool(func_tool)
        
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
        if not self.model_name or not self.provider or not self.system_prompt:
            missing = []
            if not self.model_name: missing.append("model_name")
            if not self.provider: missing.append("provider")
            if not self.system_prompt: missing.append("system_prompt or system_prompt_file")
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
        
        # Process function-based tools using the tool manager
        self._process_function_tools()
        
        # Synchronize enabled tools between self.tools["enabled"] and tool_manager.enabled_tools
        self._synchronize_enabled_tools()
                
        # Handle linked programs recursively
        self._compile_linked_programs()
        
        # Mark as compiled
        self.compiled = True
        return self
        
    def _process_function_tools(self) -> None:
        """Process function-based tools and register them."""
        if not hasattr(self, "_function_tools") or not self._function_tools:
            return
            
        # Process using the tool manager
        self.tool_manager.process_function_tools()
        
        # Ensure enabled tools list exists
        if "enabled" not in self.tools:
            self.tools["enabled"] = []
            
        # Synchronize enabled tools
        self._synchronize_enabled_tools()
            
    def _synchronize_enabled_tools(self) -> None:
        """Synchronize enabled tools between self.tools["enabled"] and tool_manager.enabled_tools."""
        # First ensure tools["enabled"] exists
        if "enabled" not in self.tools:
            self.tools["enabled"] = []
            
        # Add any tools from tool_manager.enabled_tools that aren't in self.tools["enabled"]
        for tool_name in self.tool_manager.enabled_tools:
            if tool_name not in self.tools["enabled"]:
                self.tools["enabled"].append(tool_name)
                
        # Update the tool_manager's enabled_tools with any in self.tools["enabled"]
        for tool_name in self.tools["enabled"]:
            if tool_name not in self.tool_manager.enabled_tools:
                self.tool_manager.enabled_tools.append(tool_name)
            
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
                    warnings.warn(f"Linked program not found: {program_or_path}", stacklevel=2)
            elif isinstance(program_or_path, LLMProgram):
                # It's already a program instance, compile it if not already compiled
                if not program_or_path.compiled:
                    program_or_path._compile_self()
                compiled_linked[name] = program_or_path
            else:
                raise ValueError(f"Invalid linked program type for {name}: {type(program_or_path)}")
                
        # Replace linked_programs with compiled versions
        self.linked_programs = compiled_linked
        
    def add_linked_program(self, name: str, program: "LLMProgram", description: str = "") -> "LLMProgram":
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
        
    def add_preload_file(self, file_path: str) -> "LLMProgram":
        """Add a file to preload into the system prompt.
        
        Args:
            file_path: Path to the file to preload
            
        Returns:
            self (for method chaining)
        """
        self.preload_files.append(file_path)
        return self
        
    def configure_env_info(self, variables: list[str] | str = "all") -> "LLMProgram":
        """Configure environment information sharing.
        
        This method configures which environment variables will be included in the
        system prompt for added context. For privacy/security, this is disabled by default.
        
        Args:
            variables: List of variables to include, or "all" to include all standard variables
                      Standard variables include: "working_directory", "platform", "date", 
                      "python_version", "hostname", "username"
            
        Returns:
            self (for method chaining)
            
        Examples:
            ```python
            # Include specific environment variables
            program.configure_env_info(["working_directory", "platform", "date"])
            
            # Include all standard environment variables
            program.configure_env_info("all")
            
            # Explicitly disable environment information (default)
            program.configure_env_info([])
            ```
        """
        if variables == "all":
            self.env_info = {"variables": "all"}
        else:
            self.env_info = {"variables": variables}
        return self
        
    def configure_file_descriptor(self, 
                                 enabled: bool = True, 
                                 max_direct_output_chars: int = 8000,
                                 default_page_size: int = 4000,
                                 max_input_chars: int = 8000,
                                 page_user_input: bool = True,
                                 enable_references: bool = True) -> "LLMProgram":
        """Configure the file descriptor system.
        
        The file descriptor system provides Unix-like pagination for large outputs,
        allowing LLMs to handle content that would exceed context limits.
        
        Args:
            enabled: Whether to enable the file descriptor system
            max_direct_output_chars: Threshold for FD creation
            default_page_size: Page size for pagination
            max_input_chars: Threshold for user input FD creation
            page_user_input: Whether to page user input
            enable_references: Whether to enable reference ID system
            
        Returns:
            self (for method chaining)
            
        Examples:
            ```python
            # Enable with default settings
            program.configure_file_descriptor()
            
            # Configure with custom settings
            program.configure_file_descriptor(
                max_direct_output_chars=10000,
                default_page_size=5000,
                enable_references=True
            )
            
            # Disable file descriptor system
            program.configure_file_descriptor(enabled=False)
            ```
        """
        self.file_descriptor = {
            "enabled": enabled,
            "max_direct_output_chars": max_direct_output_chars,
            "default_page_size": default_page_size,
            "max_input_chars": max_input_chars,
            "page_user_input": page_user_input,
            "enable_references": enable_references
        }
        return self
        
    def configure_thinking(self, enabled: bool = True, budget_tokens: int = 4096) -> "LLMProgram":
        """Configure Claude 3.7 thinking capability.
        
        This method configures the thinking capability for Claude 3.7 models, allowing
        the model to perform deeper reasoning on complex problems.
        
        Args:
            enabled: Whether to enable thinking capability
            budget_tokens: Budget for thinking in tokens (1024-32768)
            
        Returns:
            self (for method chaining)
            
        Note:
            This only applies to Claude 3.7 models. For other models, this configuration
            will be ignored.
            
        Examples:
            ```python
            # Enable thinking with default budget
            program.configure_thinking()
            
            # Enable thinking with custom budget
            program.configure_thinking(budget_tokens=8192)
            
            # Disable thinking
            program.configure_thinking(enabled=False)
            ```
        """
        # Ensure parameters dict exists
        if self.parameters is None:
            self.parameters = {}
            
        # Configure thinking
        self.parameters["thinking"] = {
            "type": "enabled" if enabled else "disabled",
            "budget_tokens": budget_tokens
        }
        return self
        
    def enable_token_efficient_tools(self) -> "LLMProgram":
        """Enable token-efficient tool use for Claude 3.7 models.
        
        This method enables the token-efficient tools feature which can
        significantly reduce token usage when working with tools.
        
        Returns:
            self (for method chaining)
            
        Note:
            This only applies to Claude 3.7 models. For other models, this configuration
            will be ignored.
            
        Examples:
            ```python
            # Enable token-efficient tools
            program.enable_token_efficient_tools()
            ```
        """
        # Ensure parameters dict exists
        if self.parameters is None:
            self.parameters = {}
            
        # Ensure extra_headers dict exists
        if "extra_headers" not in self.parameters:
            self.parameters["extra_headers"] = {}
            
        # Add header for token-efficient tools
        self.parameters["extra_headers"]["anthropic-beta"] = "token-efficient-tools-2025-02-19"
        return self
        
    def configure_mcp(self, config_path: str, tools: dict[str, list[str] | str] = None) -> "LLMProgram":
        """Configure Model Context Protocol (MCP) tools.
        
        This method configures MCP tool access for the program.
        
        Args:
            config_path: Path to the MCP servers configuration file
            tools: Dictionary mapping server names to lists of tools to enable,
                  or "all" to enable all tools from a server
            
        Returns:
            self (for method chaining)
            
        Examples:
            ```python
            # Enable specific tools from servers
            program.configure_mcp(
                config_path="config/mcp_servers.json",
                tools={
                    "sequential-thinking": "all",
                    "github": ["search_repositories", "get_file_contents"]
                }
            )
            
            # Enable only MCP configuration without tools
            program.configure_mcp(config_path="config/mcp_servers.json")
            ```
        """
        self.mcp_config_path = config_path
        if tools:
            self.mcp_tools = tools
        return self
        
    def add_tool(self, tool) -> "LLMProgram":
        """Add a tool to this program.
        
        This method allows adding tools to the program in multiple ways:
        1. Adding a function decorated with @register_tool
        2. Adding a regular function (will be converted to a tool using its name and docstring)
        3. Adding a tool definition dictionary with name and other properties
        
        Args:
            tool: Either a function to register as a tool, or a tool definition dictionary
            
        Returns:
            self (for method chaining)
            
        Examples:
            ```python
            # Register a tool using a dictionary
            program.add_tool({"name": "my_tool", "enabled": True})
            
            # Register a function as a tool
            @register_tool(description="Searches for weather")
            def get_weather(location: str):
                # Implementation...
                return {"temperature": 22}
                
            program.add_tool(get_weather)
            
            # Register a regular function (auto-converts to tool)
            def search_docs(query: str, limit: int = 5) -> list:
                '''Search documentation for a query.
                
                Args:
                    query: The search query
                    limit: Maximum results to return
                    
                Returns:
                    List of matching documents
                '''
                # Implementation...
                return [{"title": "Doc1"}]
                
            program.add_tool(search_docs)
            ```
        """
        # Add to tool manager (checking for duplicates)
        if callable(tool):
            # Get tool name from tool metadata or function name
            tool_name = getattr(tool, "_tool_name", tool.__name__)
            
            # Check if this tool is already registered with the tool manager
            is_duplicate = False
            for existing_tool in self.tool_manager.function_tools:
                existing_name = getattr(existing_tool, "_tool_name", existing_tool.__name__)
                if existing_name == tool_name or existing_tool is tool:
                    is_duplicate = True
                    logger.debug(f"Tool {tool_name} already registered, skipping duplicate")
                    break
                    
            if not is_duplicate:
                self.tool_manager.add_function_tool(tool)
                
                # Store in _function_tools list for tool processing
                if not hasattr(self, "_function_tools"):
                    self._function_tools = []
                self._function_tools.append(tool)
                
        elif isinstance(tool, dict):
            # For dictionary-based tools
            if "name" in tool:
                tool_name = tool["name"]
                # Check for duplicate in enabled tools
                if "enabled" in self.tools and tool_name in self.tools["enabled"]:
                    logger.debug(f"Tool {tool_name} already in enabled tools, skipping duplicate")
                else:
                    self.tool_manager.add_dict_tool(tool)
                    
                    # Make sure we have an enabled list for tools
                    if "enabled" not in self.tools:
                        self.tools["enabled"] = []
                    if tool["name"] not in self.tools["enabled"]:
                        self.tools["enabled"].append(tool["name"])
            else:
                # No name to check for duplicates, add it
                self.tool_manager.add_dict_tool(tool)
        else:
            # Invalid tool type
            raise ValueError(f"Invalid tool type: {type(tool)}. Expected callable or dict.")
        
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

    async def start(self) -> "LLMProcess":  # noqa: F821
        """Create and fully initialize an LLMProcess from this program."""
        # Ensure compiled
        if not self.compiled:
            self.compile()
            
        # Create process and set up linked programs
        process = await llmproc.LLMProcess.create(program=self)
        
        if self.linked_programs:
            process.has_linked_programs = True
            
        if self.linked_program_descriptions:
            process.linked_program_descriptions = self.linked_program_descriptions

        return process
        