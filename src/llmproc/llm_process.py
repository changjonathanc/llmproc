"""LLMProcess class for handling LLM interactions."""

import asyncio
import json
import logging
import os
import tomllib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from llmproc.program import LLMProgram

try:
    from mcp_registry import MCPAggregator, ServerRegistry, get_config_path

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from llmproc.providers import get_provider_client

from llmproc.tools import spawn_tool, fork_tool, spawn_tool_def, fork_tool_def
# New
from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor

try:
    from llmproc.providers.anthropic_tools import (
        dump_api_error,
        run_anthropic_with_tools,
    )

    HAS_ANTHROPIC_TOOLS = True
except ImportError:
    HAS_ANTHROPIC_TOOLS = False

load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)


class LLMProcess:
    """Process for interacting with LLMs using standardized program definitions."""

    def __init__(
        self,
        program: "LLMProgram",
        linked_programs_instances: dict[str, "LLMProcess"] | None = None,
    ) -> None:
        """Initialize LLMProcess from a compiled program.

        Args:
            program: A compiled LLMProgram instance
            linked_programs_instances: Dictionary of pre-initialized LLMProcess instances

        Raises:
            NotImplementedError: If the provider is not implemented
            ImportError: If the required package for a provider is not installed
            FileNotFoundError: If required files (system prompt file, MCP config file) cannot be found
            ValueError: If MCP is enabled but provider is not anthropic

        Notes:
            Missing preload files will generate warnings but won't cause the initialization to fail
        """
        # Store the program reference
        self.program = program

        # Extract core attributes from program
        self.model_name = program.model_name
        self.provider = program.provider
        self.system_prompt = program.system_prompt  # Basic system prompt without enhancements
        self.display_name = program.display_name
        self.config_dir = program.base_dir  # Use base_dir from program
        self.api_params = program.api_params
        self.parameters = {} # Keep empty - parameters are already processed in program

        # Initialize state for preloaded content
        self.preloaded_content = {}

        # Track the enriched system prompt (will be set on first run)
        self.enriched_system_prompt = None

        # Extract tool configuration
        self.enabled_tools = []
        if hasattr(program, 'tools') and program.tools:
            # Get enabled tools from the program
            self.enabled_tools = program.tools.get("enabled", [])

        # Initialize tool-related attributes
        self.tools = []
        self.tool_handlers = {}

        # MCP Configuration
        self.mcp_enabled = False
        self.mcp_config_path = getattr(program, "mcp_config_path", None)
        self.mcp_tools = getattr(program, "mcp_tools", {})
        self._mcp_initialized = False

        # Linked Programs Configuration
        self.linked_programs = {}
        self.has_linked_programs = False

        # Initialize linked programs if provided in constructor
        if linked_programs_instances:
            self.has_linked_programs = True
            self.linked_programs = linked_programs_instances
        # Otherwise use linked programs from the program - store as program objects
        elif hasattr(program, "linked_programs") and program.linked_programs:
            self.has_linked_programs = True
            self.linked_programs = program.linked_programs

        # Call the method to initialize all tools (both MCP and system tools)
        self._initialize_tools()

        # Get project_id and region for Vertex if provided in parameters
        project_id = getattr(program, "project_id", None)
        region = getattr(program, "region", None)

        # Initialize the client
        self.client = get_provider_client(self.provider, self.model_name, project_id, region)

        # Store the original system prompt before any files are preloaded
        self.original_system_prompt = self.system_prompt

        # Initialize message state (will set system message on first run)
        self.state = []

        # Preload files if specified
        if hasattr(program, "preload_files") and program.preload_files:
            self.preload_files(program.preload_files)

    def preload_files(self, file_paths: list[str]) -> None:
        """Preload files and add their content to the preloaded_content dictionary.

        This method loads file content into memory but does not modify the state.
        The enriched system prompt with preloaded content will be generated on first run.
        Missing files will generate warnings but won't cause errors.

        Args:
            file_paths: List of file paths to preload
        """
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                # Issue a clear warning with both specified and resolved paths
                warnings.warn(f"Preload file not found - Specified: '{file_path}', Resolved: '{os.path.abspath(file_path)}'")
                continue

            content = path.read_text()
            self.preloaded_content[str(path)] = content

        # Reset the enriched system prompt if it was already generated
        # so it will be regenerated with the new preloaded content
        if self.enriched_system_prompt is not None:
            self.enriched_system_prompt = None

    @classmethod
    def from_toml(cls, toml_path: str | Path) -> "LLMProcess":
        """Create an LLMProcess from a TOML program file.

        This method compiles the main program and all linked programs recursively,
        then instantiates an LLMProcess with access to all linked programs as Program objects.
        Linked programs are only compiled, not instantiated as processes, until they are needed.

        Args:
            toml_path: Path to the TOML program file

        Returns:
            An initialized LLMProcess instance with access to linked programs

        Raises:
            ValueError: If program configuration is invalid
            FileNotFoundError: If the TOML file cannot be found
        """
        # Import the compiler
        from llmproc.program import LLMProgram

        # Compile the main program and all linked programs
        # This will create a properly linked object graph with Program objects
        main_path = Path(toml_path).resolve()
        main_program = LLMProgram.compile(main_path, include_linked=True, check_linked_files=True)

        if not main_program:
            raise ValueError(f"Failed to compile program from {toml_path}")

        # Create the main process
        main_process = cls(program=main_program)

        # The linked_programs attribute of main_program should now contain references to
        # compiled Program objects instead of path strings
        if hasattr(main_program, 'linked_programs') and main_program.linked_programs:
            main_process.has_linked_programs = bool(main_program.linked_programs)

        # No need to initialize spawn tool here - it's already done in the constructor
        return main_process


    async def run(self, user_input: str, max_iterations: int = 10) -> str:
        """Run the LLM process with user input asynchronously.

        This method supports full tool execution with proper async handling.
        If used in a synchronous context, it will automatically run in a new event loop.

        Args:
            user_input: The user message to process
            max_iterations: Maximum number of tool-calling iterations

        Returns:
            The model's response as a string
        """
        # Check if we're in an event loop
        try:
            asyncio.get_running_loop()
            in_event_loop = True
        except RuntimeError:
            in_event_loop = False

        # If not in an event loop, run in a new one
        if not in_event_loop:
            return asyncio.run(self._async_run(user_input, max_iterations))
        else:
            return await self._async_run(user_input, max_iterations)

    async def _async_run(self, user_input: str, max_iterations: int = 10) -> str:
        """Internal async implementation of run.

        Args:
            user_input: The user message to process
            max_iterations: Maximum number of tool-calling iterations

        Returns:
            The model's response as a string

        Raises:
            ValueError: If user_input is empty
        """
        # Verify user input isn't empty
        if not user_input or user_input.strip() == "":
            raise ValueError("User input cannot be empty")

        # Generate enriched system prompt on first run
        if self.enriched_system_prompt is None:
            self.enriched_system_prompt = self.program.get_enriched_system_prompt(
                process_instance=self,
                include_env=True
            )
            # Set the system message with enriched prompt
            self.state = [{"role": "system", "content": self.enriched_system_prompt}]

        # Add user input to state
        self.state.append({"role": "user", "content": user_input})

        self.messages = self.state # TODO; deprecate state in favor of messages, it's more descriptive

        # Create provider-specific API calls
        if self.provider == "openai":
            raise NotImplementedError("OpenAI is not yet implemented")
            pass
        elif self.provider == "anthropic":
            return await self._run_anthropic_with_tools(max_iterations)
        elif self.provider == "vertex":
            return NotImplementedError("OpenAI is not yet implemented")

        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")

    async def _run_anthropic_with_tools(self, max_iterations: int = 10) -> str:
        """Run Anthropic with tool support.

        Handles multiple iterations of tool calls and responses in an asynchronous context,
        processing each tool and providing the result back to the model.

        Args:
            max_iterations: Maximum number of tool-calling iterations

        Returns:
            The final model response as a string
        """
        if not HAS_ANTHROPIC_TOOLS:
            raise ImportError(
                "Anthropic tools support requires the llmproc.providers.anthropic_tools module."
            )

        # Extract system prompt and messages
        system_prompt = None
        messages = []

        for msg in self.state:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                # Skip empty messages that would cause API errors
                if msg.get("content") != "":
                    messages.append(msg)

        # Run the tool interaction loop through the specialized module
        # Just pass the LLMProcess instance and let the function access what it needs
        final_response = await run_anthropic_with_tools(
            llm_process=self,
            system_prompt=system_prompt,
            messages=messages,
            max_iterations=max_iterations
        )

        # Add the final response to the permanent state
        self.state.append({"role": "assistant", "content": final_response})

        return final_response

    def get_state(self) -> list[dict[str, str]]:
        """Return the current conversation state.

        Returns:
            A copy of the current conversation state
        """
        return self.state.copy()

    async def _initialize_mcp_tools_if_needed(self) -> None:
        """Initialize MCP registry and tools if they haven't been initialized yet.

        This method safely handles initialization in both synchronous and asynchronous contexts.
        It's designed to work correctly when called from both __init__ and async methods like fork_process.
        """
        # Return immediately if already initialized or MCP not enabled
        if (hasattr(self, '_mcp_initialized') and self._mcp_initialized) or not self.mcp_enabled:
            return
            
        try:
            # Initialize MCP tools
            await self._initialize_mcp_tools()
            # Mark as successfully initialized
            self._mcp_initialized = True
        except Exception as e:
            # Log the error but mark as initialized to avoid repeated attempts
            logger.warning(f"Failed to initialize MCP tools: {str(e)}")
            self._mcp_initialized = True
            # Re-raise if this is a critical error
            if isinstance(e, (ImportError, FileNotFoundError)):
                raise

    async def _initialize_mcp_tools(self) -> None:
        """Initialize MCP registry and tools.

        This sets up the MCP registry and filters tools based on user configuration.
        Only tools explicitly specified in the mcp_tools configuration will be enabled.
        Only servers that have tools configured will be initialized.
        """
        if not self.mcp_enabled:
            return

        # Ensure tools list is initialized
        if not hasattr(self, 'tools') or self.tools is None:
            self.tools = []

        # Initialize registry and aggregator
        try:
            self.registry = ServerRegistry.from_config(self.mcp_config_path)
            self.aggregator = MCPAggregator(self.registry)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MCP registry: {str(e)}")

        # Get tools grouped by server
        try:
            server_tools_map = await self.aggregator.list_tools(return_server_mapping=True)
        except Exception as e:
            raise RuntimeError(f"Failed to list MCP tools: {str(e)}")

        # Track registered tools to avoid duplicates
        registered_tools = set()
        
        # Initialize tool_handlers if not already present
        if not hasattr(self, 'tool_handlers'):
            self.tool_handlers = {}

        # Process each server and tool configuration
        for server_name, tool_config in self.mcp_tools.items():
            # Skip servers that don't exist in the available tools
            if server_name not in server_tools_map:
                logger.warning(f"Server '{server_name}' not found in available tools")
                continue

            server_tools = server_tools_map[server_name]
            
            # Create a mapping of tool names to tools for this server
            server_tool_map = {tool.name: tool for tool in server_tools}

            # Case 1: Register all tools for a server
            if tool_config == "all":
                for tool in server_tools:
                    namespaced_name = f"{server_name}__{tool.name}"
                    if namespaced_name not in registered_tools:
                        # Add to schema list for API
                        self.tools.append(self._format_tool_for_anthropic(tool, server_name))
                        
                        # Create handler function for this tool that calls the MCP aggregator
                        self.tool_handlers[namespaced_name] = self._create_mcp_tool_handler(namespaced_name)
                        
                        registered_tools.add(namespaced_name)
                
                logger.info(f"Registered all tools ({len(server_tools)}) from server '{server_name}'")

            # Case 2: Register specific tools
            elif isinstance(tool_config, list):
                for tool_name in tool_config:
                    if tool_name in server_tool_map:
                        tool = server_tool_map[tool_name]
                        namespaced_name = f"{server_name}__{tool.name}"
                        if namespaced_name not in registered_tools:
                            # Add to schema list for API
                            self.tools.append(self._format_tool_for_anthropic(tool, server_name))
                            
                            # Create handler function for this tool that calls the MCP aggregator
                            self.tool_handlers[namespaced_name] = self._create_mcp_tool_handler(namespaced_name)
                            
                            registered_tools.add(namespaced_name)
                    else:
                        logger.warning(f"Tool '{tool_name}' not found for server '{server_name}'")
                
                logger.info(f"Registered {len([t for t in tool_config if t in server_tool_map])} tools from server '{server_name}'")
        
        # Summarize registered tools
        if not self.tools:
            logger.warning("No MCP tools were registered. Check your configuration.")
        else:
            logger.info(f"Total MCP tools registered: {len(self.tools)}")
                
    def _create_mcp_tool_handler(self, tool_name: str):
        """Create a handler function for an MCP tool.
        
        Args:
            tool_name: The namespaced tool name (server__tool)
            
        Returns:
            A callable that takes tool arguments and returns the result
        """
        # Return a handler function that closes over the tool_name and self reference
        async def handler(args):
            if not self.aggregator:
                raise RuntimeError(f"MCP aggregator not initialized. Cannot call tool: {tool_name}")
                
            try:
                # Call the tool through the aggregator
                result = await self.aggregator.call_tool(tool_name, args)
                    
                return result
            except Exception as e:
                error_msg = f"Error executing MCP tool {tool_name}: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg, "is_error": True}
                
        return handler

    def _initialize_linked_programs(self, linked_programs: dict[str, Path | str]) -> None:
        """Initialize linked LLM programs from their TOML program files.

        This method compiles all linked programs recursively and stores them as Program objects
        that can be instantiated as processes when needed. This is more memory efficient and
        follows compilation semantics where programs are validated first before instantiation.

        Args:
            linked_programs: Dictionary mapping program names to TOML program paths
        """
        from llmproc.program import LLMProgram

        # Dictionary to store compiled program objects
        compiled_program_objects = {}

        # Resolve paths and compile all programs
        for program_name, program_path in linked_programs.items():
            path = Path(program_path)
            original_path = program_path

            # If path is relative and we have a config_dir, resolve it
            if not path.is_absolute() and self.config_dir:
                path = self.config_dir / path

            if not path.exists():
                raise FileNotFoundError(f"Linked program file not found - Specified: '{original_path}', Resolved: '{path}'")

            try:
                # Compile this linked program and its sub-linked programs
                # But store the programs as objects, don't instantiate them as processes yet
                compiled_program = LLMProgram.compile(path, include_linked=False)
                compiled_program_objects[program_name] = compiled_program

                # Log successful compilation
                logger.info(f"Compiled linked program '{program_name}' ({compiled_program.provider} {compiled_program.model_name})")
            except Exception as e:
                raise RuntimeError(f"Failed to compile linked program '{program_name}': {str(e)}") from e

        # Store the compiled program objects
        self.linked_programs = compiled_program_objects
        self.has_linked_programs = bool(compiled_program_objects)

    def _register_spawn_tool(self) -> None:
        """Register the spawn system call for creating new processes from linked programs."""

        # Only register if we have linked programs
        if not self.linked_programs:
            logger.warning("No linked programs available. Spawn system call not registered.")
            return

        # Create a copy of the tool definition with dynamic available programs info
        api_tool_def = spawn_tool_def.copy()
        available_programs = ", ".join(self.linked_programs.keys())
        api_tool_def["description"] = spawn_tool_def["description"] + f"\n\nAvailable programs: {available_programs}"

        # Create the handler function for the spawn tool
        async def spawn_handler(args):
            return spawn_tool(
                program_name=args.get("program_name"),
                query=args.get("query"),
                llm_process=self
            )

        # Register the handler in the unified tool_handlers dictionary
        self.tool_handlers["spawn"] = spawn_handler

        # Add the tool definition to the tools list for API
        self.tools.append(api_tool_def)
        
        logger.info(f"Registered spawn tool with access to programs: {', '.join(self.linked_programs.keys())}")

    def _register_fork_tool(self) -> None:
        """Register the fork system call for creating copies of the current process."""
        # Create a copy of the tool definition for the API
        api_tool_def = fork_tool_def.copy()
        
        # Create the handler function for the fork tool
        async def fork_handler(args):
            # The actual fork implementation is in anthropic_tools.py
            # This is just a placeholder for the handler interface
            return f"Fork command received with args: {args}"
            
        # Register the handler in the unified tool_handlers dictionary
        self.tool_handlers["fork"] = fork_handler
        
        # Add to the tools list for API
        self.tools.append(api_tool_def)
        logger.info(f"Registered fork tool for process {self.model_name}")

    def _format_tool_for_anthropic(self, tool, server_name=None):
        """Format a tool for Anthropic API.

        Args:
            tool: Tool object from MCP registry
            server_name: Optional server name for proper namespacing

        Returns:
            Dictionary with tool information formatted for Anthropic API
        """
        # Create namespaced name with server prefix
        namespaced_name = f"{server_name}__{tool.name}" if server_name else tool.name
        
        # Ensure input schema has required fields
        input_schema = tool.inputSchema.copy() if tool.inputSchema else {}
        if "type" not in input_schema:
            input_schema["type"] = "object"
        if "properties" not in input_schema:
            input_schema["properties"] = {}
            
        # Create the tool definition
        return {
            "name": namespaced_name,
            "description": tool.description,
            "input_schema": input_schema
        }


    def reset_state(
        self, keep_system_prompt: bool = True, keep_preloaded: bool = True
    ) -> None:
        """Reset the conversation state.

        Args:
            keep_system_prompt: Whether to keep the system prompt in the state
            keep_preloaded: Whether to keep preloaded file content in the state
        """
        # Clear the state
        self.state = []

        # Handle preloaded content
        if not keep_preloaded:
            # Clear preloaded content
            self.preloaded_content = {}

        # Always reset the enriched system prompt - it will be regenerated on next run
        # with the correct combination of system prompt and preloaded content
        self.enriched_system_prompt = None

    def _initialize_tools(self) -> None:
        """Initialize all tools - both MCP and system tools.
        
        This method handles the setup of tool schemas and handlers in a unified way,
        regardless of whether they are MCP tools or system tools.
        """
        # Initialize MCP if configured
        if self.mcp_config_path and self.mcp_tools:
            if not HAS_MCP:
                raise ImportError(
                    "MCP features require the mcp-registry package. Install it with 'pip install mcp-registry'."
                )

            # Currently only support Anthropic with MCP
            if self.provider != "anthropic":
                raise ValueError(
                    "MCP features are currently only supported with the Anthropic provider"
                )

            self.mcp_enabled = True
            
            # Check if we're already in an event loop
            try:
                asyncio.get_running_loop()
                # If in event loop, don't initialize now - will be done lazily
                # during the first call to run()
            except RuntimeError:
                # No event loop, create one for initialization
                asyncio.run(self._initialize_mcp_tools_if_needed())
        
        # Register system tools
        if "spawn" in self.enabled_tools and self.has_linked_programs:
            self._register_spawn_tool()

        if "fork" in self.enabled_tools:
            self._register_fork_tool()

    async def call_tool(self, tool_name: str, args: dict) -> Any:
        """Call a tool by name with the given arguments.
        
        This method provides a unified interface for calling any registered tool,
        whether it's an MCP tool or a system tool like spawn or fork.
        
        Args:
            tool_name: The name of the tool to call
            args: The arguments to pass to the tool
            
        Returns:
            The result of the tool execution
            
        Raises:
            ValueError: If the tool is not found
            RuntimeError: If the tool execution fails
        """
        if not hasattr(self, 'tool_handlers') or not self.tool_handlers:
            raise ValueError("No tool handlers registered")
            
        if tool_name not in self.tool_handlers:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {', '.join(self.tool_handlers.keys())}")
            
        try:
            handler = self.tool_handlers[tool_name]
            return await handler(args)
        except Exception as e:
            raise RuntimeError(f"Error executing tool '{tool_name}': {str(e)}")
    
    async def fork_process(self) -> "LLMProcess":
        """Create a deep copy of this process with preserved state.

        This implements the fork system call semantics where a copy of the
        process is created with the same state and configuration. The forked
        process is completely independent and can run separate tasks.

        Returns:
            A new LLMProcess instance that is a deep copy of this one
        """
        import copy

        # Create a new instance of LLMProcess with the same program
        forked_process = LLMProcess(program=self.program)

        # Copy the enriched system prompt if it exists
        if hasattr(self, 'enriched_system_prompt') and self.enriched_system_prompt:
            forked_process.enriched_system_prompt = self.enriched_system_prompt

        # Deep copy the conversation state
        forked_process.state = copy.deepcopy(self.state)

        # Copy any preloaded content
        if hasattr(self, 'preloaded_content') and self.preloaded_content:
            forked_process.preloaded_content = copy.deepcopy(self.preloaded_content)

        # If the forked process has MCP enabled, make sure it's initialized properly
        if forked_process.mcp_enabled and hasattr(forked_process, '_mcp_initialized') and not forked_process._mcp_initialized:
            await forked_process._initialize_mcp_tools_if_needed()

        # Preserve any other state we need
        # Note: We don't copy tool handlers as they're already set up in the constructor

        return forked_process
