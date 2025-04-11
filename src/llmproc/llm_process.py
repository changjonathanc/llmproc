"""LLMProcess class for executing LLM programs and handling interactions."""

import asyncio
import copy
import logging
import os
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.program import LLMProgram
from llmproc.providers import get_provider_client
from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor
from llmproc.providers.constants import ANTHROPIC_PROVIDERS, GEMINI_PROVIDERS
from llmproc.providers.gemini_process_executor import GeminiProcessExecutor
from llmproc.providers.openai_process_executor import OpenAIProcessExecutor
from llmproc.common.results import RunResult, ToolResult
from llmproc.tools import ToolManager, file_descriptor_instructions

# Check if mcp-registry is installed
HAS_MCP = False
try:
    import mcp_registry  # noqa

    HAS_MCP = True
except ImportError:
    pass

load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)


class LLMProcess:
    """Process for interacting with LLMs using standardized program definitions."""

    def __init__(
        self,
        program: LLMProgram,
        linked_programs_instances: dict[str, "LLMProcess"] | None = None,
        skip_tool_init: bool = False,
    ) -> None:
        """Initialize LLMProcess from a compiled program.
        
        ⚠️ WARNING: DO NOT USE THIS CONSTRUCTOR DIRECTLY! ⚠️
        
        ALWAYS use the async factory method `await program.start()` instead, which properly
        handles initialization following the Unix-inspired pattern:
        
        ```python
        program = LLMProgram.from_toml("config.toml")
        process = await program.start()  # CORRECT WAY TO CREATE PROCESS
        ```
        
        Direct instantiation with `LLMProcess(program=...)` will cause issues with:
        - Context-aware tools (spawn, goto, fd_tools)
        - Runtime dependency injection
        - MCP tool registration
        - Proper initialization order
        
        Args:
            program: A compiled LLMProgram instance
            linked_programs_instances: Dictionary of pre-initialized LLMProcess instances
            skip_tool_init: Internal flag to skip tool initialization (for use with factory methods)

        Raises:
            NotImplementedError: If the provider is not implemented
            ImportError: If the required package for a provider is not installed
            FileNotFoundError: If required files (system prompt file, MCP config file) cannot be found
            ValueError: If MCP is enabled but provider is not anthropic

        Notes:
            This constructor exists primarily for internal use and testing.
            For all production code, ALWAYS use `await program.start()` instead.
            Direct instantiation will likely result in broken context-aware tools.
        """
        # Store the program reference
        self.program = program
        
        # Use the common initialization method
        self._initialize_from_program(program, linked_programs_instances, skip_tool_init)

    @classmethod
    async def create(
        cls,
        program: LLMProgram,
        linked_programs_instances: dict[str, "LLMProcess"] | None = None,
    ) -> "LLMProcess":
        """Create and fully initialize an LLMProcess asynchronously.
        
        ⚠️ INTERNAL API - DO NOT USE DIRECTLY ⚠️
        
        ALWAYS use LLMProgram.start() instead of this factory method:
        
        ```python
        program = LLMProgram.from_toml("config.toml")
        process = await program.start()  # Correct pattern - never use LLMProcess.create()
        ```
        
        This factory method implements the Unix-inspired initialization approach (RFC053):
        1. Get tool configuration from program
        2. Initialize tools with configuration (without process)
        3. Create process with pre-initialized tools
        4. Runtime context is automatically set up during initialization

        Args:
            program: The LLMProgram to use
            linked_programs_instances: Dictionary of pre-initialized LLMProcess instances

        Returns:
            A fully initialized LLMProcess

        Raises:
            All exceptions from __init__, plus:
            RuntimeError: If MCP initialization fails
            ValueError: If a server specified in mcp_tools is not found in available tools
            
        Notes:
            This is an implementation detail used by LLMProgram.start() and should not be
            called directly by users. It exists to maintain a clean separation between
            the configuration (program) and runtime (process) phases.
        """
        # Phase 1: Get tool configuration from program
        # This is the configuration-based approach from RFC053 that avoids circular dependencies
        tool_config = program.get_tool_configuration(linked_programs_instances)
        
        # Phase 2: Initialize tools with configuration (without process)
        # This avoids circular dependencies between process and tools
        await program.tool_manager.initialize_tools(tool_config)
        
        # Phase 3: Create process instance with pre-initialized tools
        # Skip tool initialization in constructor since we've already initialized them
        instance = cls(program, linked_programs_instances, skip_tool_init=True)
                
        # Check for any deferred tool initializations
        if hasattr(instance, "_tools_need_initialization") and instance._tools_need_initialization:
            logger.info("Running remaining deferred tool initializations")
            instance._tools_need_initialization = False

        return instance
        

    def _setup_runtime_context(self) -> None:
        """Set up runtime context for tool execution.
        
        This helper method creates a runtime context dictionary containing all dependencies
        needed by context-aware tools, and sets it on the tool manager. Tools that are
        decorated with @context_aware will receive this context at runtime.
        """
        runtime_context = {
            "process": self,
            "fd_manager": self.fd_manager,
            "linked_programs": self.linked_programs,
            "linked_program_descriptions": self.linked_program_descriptions,
        }
        self.tool_manager.set_runtime_context(runtime_context)
        logger.debug(f"Runtime context set with keys: {', '.join(runtime_context.keys())}")
    
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
                warnings.warn(
                    f"Preload file not found - Specified: '{file_path}', Resolved: '{os.path.abspath(file_path)}'",
                    stacklevel=2,
                )
                continue

            content = path.read_text()
            self.preloaded_content[str(path)] = content

        # Reset the enriched system prompt if it was already generated
        # so it will be regenerated with the new preloaded content
        if self.enriched_system_prompt is not None:
            self.enriched_system_prompt = None

    async def run(self, user_input: str, max_iterations: int = 10, callbacks: dict = None) -> "RunResult":
        """Run the LLM process with user input asynchronously.

        This method supports full tool execution with proper async handling.
        If used in a synchronous context, it will automatically run in a new event loop.

        Args:
            user_input: The user message to process
            max_iterations: Maximum number of tool-calling iterations
            callbacks: Optional dictionary of callback functions:
                - 'on_tool_start': Called when a tool execution starts
                - 'on_tool_end': Called when a tool execution completes
                - 'on_response': Called when a model response is received

        Returns:
            RunResult object with execution metrics
        """
        # Check if we're in an event loop
        try:
            asyncio.get_running_loop()
            in_event_loop = True
        except RuntimeError:
            in_event_loop = False

        # If not in an event loop, run in a new one
        if not in_event_loop:
            return asyncio.run(self._async_run(user_input, max_iterations, callbacks))
        else:
            return await self._async_run(user_input, max_iterations, callbacks)

    async def _async_run(self, user_input: str, max_iterations: int = 10, callbacks: dict = None) -> "RunResult":
        """Internal async implementation of run.

        Args:
            user_input: The user message to process
            max_iterations: Maximum number of tool-calling iterations
            callbacks: Optional dictionary of callback functions

        Returns:
            RunResult object with execution metrics

        Raises:
            ValueError: If user_input is empty
        """
        # Create a RunResult object to track this run
        run_result = RunResult()

        # Normalize callbacks
        callbacks = callbacks or {}

        # Verify user input isn't empty
        if not user_input or user_input.strip() == "":
            raise ValueError("User input cannot be empty")

        # Check if tools need initialization (happens if LLMProcess was created within an event loop)
        if hasattr(self, "_tools_need_initialization") and self._tools_need_initialization:
            logger.info("Initializing tools before first run (deferred from __init__)")
            await self._initialize_tools()
            self._tools_need_initialization = False

        # MCP tools should already be initialized during program.start()
        # No need for lazy initialization here

        # Generate enriched system prompt on first run
        if self.enriched_system_prompt is None:
            self.enriched_system_prompt = self.program.get_enriched_system_prompt(process_instance=self, include_env=True)

        # Process user input through file descriptor manager if enabled
        processed_user_input = user_input
        if self.file_descriptor_enabled:  # fd_manager is guaranteed to exist if enabled flag is true
            # Delegate to file descriptor manager to handle large user input
            processed_user_input = self.fd_manager.handle_user_input(user_input)

            # Log if input was converted to a file descriptor
            if processed_user_input != user_input:
                logger.info(f"Large user input ({len(user_input)} chars) converted to file descriptor")

        # Add processed user input to state
        self.state.append({"role": "user", "content": processed_user_input})

        # Create provider-specific process executors
        if self.provider == "openai":
            # Use the OpenAI process executor (simplified version)
            executor = OpenAIProcessExecutor()
            run_result = await executor.run(self, user_input, max_iterations, callbacks, run_result)

        elif self.provider in ANTHROPIC_PROVIDERS:
            # Use the stateless AnthropicProcessExecutor for both direct Anthropic API and Vertex AI
            executor = AnthropicProcessExecutor()
            run_result = await executor.run(self, user_input, max_iterations, callbacks, run_result)

        elif self.provider in GEMINI_PROVIDERS:
            # Use the GeminiProcessExecutor for both direct API and Vertex AI
            executor = GeminiProcessExecutor()
            run_result = await executor.run(self, user_input, max_iterations, callbacks, run_result)
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")

        # Process any references in the last assistant message for reference ID system
        if self.file_descriptor_enabled:  # fd_manager is guaranteed to exist if enabled flag is true
            # Get the last assistant message if available
            if self.state and len(self.state) > 0 and self.state[-1].get("role") == "assistant":
                assistant_message = self.state[-1].get("content", "")

                # Check if we have a string message or a structured message (Anthropic)
                if isinstance(assistant_message, list):
                    # Process each text block in the message
                    for _i, block in enumerate(assistant_message):
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content = block.get("text", "")
                            # Delegate to the FileDescriptorManager
                            self.fd_manager.process_references(text_content)
                else:
                    # Process the simple string message - delegate to the FileDescriptorManager
                    self.fd_manager.process_references(assistant_message)

        # Mark the run as complete and calculate duration
        run_result.complete()

        return run_result

    def get_state(self) -> list[dict[str, str]]:
        """Return the current conversation state.

        Returns:
            A copy of the current conversation state
        """
        return self.state.copy()

    # _initialize_mcp_tools method has been removed.
    # MCP initialization is now handled entirely by the ToolManager
    # as part of the _initialize_tools method, which delegates to
    # ToolManager.initialize_tools.

    def reset_state(self, keep_system_prompt: bool = True, keep_preloaded: bool = True, keep_file_descriptors: bool = True) -> None:
        """Reset the conversation state.

        Args:
            keep_system_prompt: Whether to keep the system prompt for the next API call
            keep_preloaded: Whether to keep preloaded file content
            keep_file_descriptors: Whether to keep file descriptor content

        Note:
            State only contains user/assistant messages, not system message.
            System message is stored separately in enriched_system_prompt.
        """
        # Clear the conversation state (user/assistant messages)
        self.state = []

        # Handle preloaded content
        if not keep_preloaded:
            # Clear preloaded content
            self.preloaded_content = {}

        # If we're not keeping the system prompt, reset it to original
        if not keep_system_prompt:
            self.system_prompt = self.original_system_prompt

        # Reset file descriptors if not keeping them
        if not keep_file_descriptors and self.file_descriptor_enabled and self.fd_manager:
            # Create a new manager but preserve the settings
            self.fd_manager = FileDescriptorManager(
                default_page_size=self.fd_manager.default_page_size,
                max_direct_output_chars=self.fd_manager.max_direct_output_chars,
                max_input_chars=self.fd_manager.max_input_chars,
                page_user_input=self.fd_manager.page_user_input,
            )
            # Copy over the FD-related tools registry
            self.fd_manager.fd_related_tools = self.fd_manager.fd_related_tools.union(self.fd_manager._FD_RELATED_TOOLS)

        # Always reset the enriched system prompt - it will be regenerated on next run
        # with the correct combination of system prompt and preloaded content
        self.enriched_system_prompt = None

    async def _initialize_tools(self) -> None:
        """Initialize all system tools.

        This method uses the configuration-based approach to handle tool initialization,
        which avoids circular dependencies between LLMProcess and tools. The steps are:
        
        1. Get tool configuration from the program
        2. Delegate to the tool_manager.initialize_tools method for all registration
        
        This two-step approach follows the Unix-inspired initialization pattern (RFC053).

        Returns:
            None

        Raises:
            ImportError: If MCP is enabled but mcp-registry package is not installed
            ValueError: If MCP is enabled with an unsupported provider
        """
        # Basic validation for MCP tools
        if self._needs_async_init:
            if not HAS_MCP:
                raise ImportError("MCP features require the mcp-registry package. Install it with 'pip install mcp-registry'.")

            # Currently only support Anthropic with MCP
            if self.provider != "anthropic":
                raise ValueError("MCP features are currently only supported with the Anthropic provider")

        # Get tool configuration from the program
        # This extracts all necessary configuration for tools without circular references
        tool_config = self.program.get_tool_configuration()
        
        # Delegate to the ToolManager's initialize_tools method with configuration
        # This method handles loading builtin tools, registering enabled tools,
        # and initializing MCP tools if MCP is enabled
        await self.tool_manager.initialize_tools(tool_config)
        
        # Log the enabled tools
        enabled_tools = self.tool_manager.get_enabled_tools()
        logger.info(f"Initialized {len(enabled_tools)} tools using ToolManager: {', '.join(enabled_tools)}")

    @property
    def tools(self) -> list:
        """Property to access tool definitions for the LLM API.

        This delegates to the ToolManager which provides a consistent interface
        for getting tool schemas across all tool types.

        The ToolManager handles filtering, alias resolution, and validation.

        Returns:
            List of tool schemas formatted for the LLM provider's API.
        """
        # Get schemas from the tool manager
        # This includes filtering for enabled tools and alias transformation
        return self.tool_manager.get_tool_schemas()

    @property
    def tool_handlers(self) -> dict:
        """Property to access tool handler functions.

        This delegates to the ToolManager's registry to provide access to the
        actual handler functions that execute tool operations.

        Returns:
            Dictionary mapping tool names to their handler functions.
        """
        return self.tool_manager.runtime_registry.tool_handlers
        
    def _initialize_file_descriptor(self, program: LLMProgram) -> None:
        """Initialize file descriptor subsystem based on program configuration."""
        # By the time we get here, program has been compiled and dependencies between
        # FD system and tools have been resolved. If file_descriptor.enabled is True,
        # we can proceed with initialization without additional checks.
        if hasattr(program, "file_descriptor"):
            fd_config = program.file_descriptor
            enabled = fd_config.get("enabled", False)
            
            if enabled:
                # Create file descriptor manager with configuration
                self.references_enabled = fd_config.get("enable_references", False)
                self.fd_manager = FileDescriptorManager(
                    default_page_size=fd_config.get("default_page_size", 4000),
                    max_direct_output_chars=fd_config.get("max_direct_output_chars", 8000),
                    max_input_chars=fd_config.get("max_input_chars", 8000),
                    page_user_input=fd_config.get("page_user_input", True),
                    enable_references=self.references_enabled,
                )
                self.file_descriptor_enabled = True
                logger.info(f"File descriptor enabled: page_size={self.fd_manager.default_page_size}, references={self.references_enabled}")
            
    def _initialize_linked_programs(self, program: LLMProgram, linked_programs_instances: dict[str, "LLMProcess"] | None = None) -> None:
        """Initialize linked programs from provided instances or program configuration."""
        if linked_programs_instances:
            self.linked_programs = linked_programs_instances
            self.has_linked_programs = bool(linked_programs_instances)
        elif hasattr(program, "linked_programs") and program.linked_programs:
            self.linked_programs = program.linked_programs
            self.has_linked_programs = bool(program.linked_programs)
            
        if hasattr(program, "linked_program_descriptions"):
            self.linked_program_descriptions = program.linked_program_descriptions
            
    def _initialize_preloaded_content(self, program: LLMProgram) -> None:
        """Load any preloaded files from program configuration."""
        if hasattr(program, "preload_files") and program.preload_files:
            self.preload_files(program.preload_files)
            
    def _initialize_from_program(
        self, 
        program: LLMProgram, 
        linked_programs_instances: dict[str, "LLMProcess"] | None = None,
        skip_tool_init: bool = False
    ) -> None:
        """Common initialization method for both __init__ and create().
        
        This follows the Unix-inspired program/process model from RFC053, where:
        1. Program (static definition) provides the configuration
        2. Process (runtime instance) is initialized from this configuration
        3. Runtime context is established for tool execution
        
        This centralized method handles all common initialization steps to avoid
        code duplication between synchronous and asynchronous initialization paths.
        
        Args:
            program: The compiled LLMProgram to initialize from
            linked_programs_instances: Optional pre-initialized process instances  
            skip_tool_init: Whether to skip tool initialization (should be True when
                            tools are initialized before process creation in Unix pattern)
        
        Note:
            Direct usage of this method is not recommended. Use LLMProgram.start()
            for the cleanest initialization flow.
        """
        # Extract core attributes from program
        self.model_name = program.model_name
        self.provider = program.provider
        self.system_prompt = program.system_prompt
        self.display_name = program.display_name
        self.base_dir = program.base_dir
        self.api_params = program.api_params
        self.parameters = {}  # Parameters are already processed in program
        
        # Initialize state tracking
        self.preloaded_content = {}
        self.enriched_system_prompt = None
        self.original_system_prompt = self.system_prompt
        self.state = []
        self.allow_fork = True
        
        # Initialize tool configuration
        self.enabled_tools = []
        if hasattr(program, "tools") and program.tools:
            self.enabled_tools = program.tools.get("enabled", [])
        self.tool_manager = program.tool_manager
        
        # Initialize MCP configuration
        self.mcp_config_path = getattr(program, "mcp_config_path", None)
        self.mcp_tools = getattr(program, "mcp_tools", {})
        self.mcp_enabled = self.mcp_config_path is not None
        self._needs_async_init = self.mcp_config_path is not None
        
        # Initialize file descriptor subsystem
        self.file_descriptor_enabled = False
        self.fd_manager = None
        self.references_enabled = False
        
        # Initialize linked programs
        self.linked_programs = {}
        self.has_linked_programs = False
        self.linked_program_descriptions = {}
        
        # Use the helper methods for specific initialization tasks
        self._initialize_linked_programs(program, linked_programs_instances)
        self._initialize_file_descriptor(program)
        
        # Initialize client
        project_id = getattr(program, "project_id", None)
        region = getattr(program, "region", None)
        self.client = get_provider_client(self.provider, self.model_name, project_id, region)
        
        # Set up runtime context for tools
        self._setup_runtime_context()
        
        # Initialize preloaded content
        self._initialize_preloaded_content(program)
        
        # Check if OpenAI provider is used with tools (not yet supported)
        if self.provider == "openai" and self.enabled_tools:
            raise ValueError("Tool usage is not yet supported for OpenAI models in this implementation.")
            
        # Handle tool initialization based on the initialization path
        self._tools_need_initialization = False
        if not skip_tool_init:
            try:
                # Check if we're already in an event loop
                asyncio.get_running_loop()
                # If we get here, we're in an event loop, which means we can't use asyncio.run
                logger.warning("LLMProcess initialized from within an existing event loop - deferring tool initialization")
                self._tools_need_initialization = True
            except RuntimeError:
                # Not in an event loop, safe to use asyncio.run
                asyncio.run(self._initialize_tools())
        else:
            # Skip initialization when using Unix-inspired pattern (RFC053)
            logger.info("Skipping tool initialization (Unix-inspired initialization)")

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool by name with the given arguments.

        This method provides a unified interface for calling any registered tool,
        whether it's an MCP tool, a system tool, or a function-based tool.
        It delegates to the ToolManager which handles all tool calling details.

        Args:
            tool_name: The name of the tool to call
            **kwargs: The keyword arguments to pass to the tool
            
        Returns:
            The result of the tool execution or an error ToolResult
        """
        try:
            # Pass all keyword arguments directly to the tool manager
            args_dict = dict(kwargs)
            return await self.tool_manager.call_tool(tool_name, args_dict)
        except Exception as e:
            return ToolResult.from_error(f"Error calling tool '{tool_name}': {str(e)}")

    async def count_tokens(self):
        """Count tokens in the current conversation state.

        Returns:
            dict: Token count information with provider-specific details, including:
                - input_tokens: Number of tokens in conversation
                - context_window: Max tokens supported by the model
                - percentage: Percentage of context window used
                - remaining_tokens: Number of tokens left in context window
                - cached_tokens: (Gemini only) Number of tokens in cached content
                - note: Informational message (when estimation is used)
                - error: Error message if token counting failed
        """
        # Create the appropriate executor based on provider
        if self.provider in ANTHROPIC_PROVIDERS:
            executor = AnthropicProcessExecutor()
            return await executor.count_tokens(self)
        elif self.provider in GEMINI_PROVIDERS:
            executor = GeminiProcessExecutor()
            return await executor.count_tokens(self)
        else:
            # No token counting support for this provider
            return None

    def get_last_message(self) -> str:
        """Get the most recent message from the conversation.

        Returns:
            The text content of the last assistant message,
            or an empty string if the last message is not from an assistant.

        Note:
            This handles both string content and structured content blocks from
            providers like Anthropic.
        """
        # Check if state has any messages
        if not self.state:
            return ""

        # Get the last message
        last_message = self.state[-1]

        # Return content if it's an assistant message, empty string otherwise
        if last_message.get("role") == "assistant" and "content" in last_message:
            content = last_message["content"]

            # If content is a string, return it directly
            if isinstance(content, str):
                return content

            # Handle Anthropic's content blocks format
            if isinstance(content, list):
                extracted_text = []
                for block in content:
                    # Handle text blocks
                    if isinstance(block, dict) and block.get("type") == "text":
                        extracted_text.append(block.get("text", ""))
                    # Handle TextBlock objects which may be used by Anthropic
                    elif hasattr(block, "text") and hasattr(block, "type"):
                        if block.type == "text":
                            extracted_text.append(getattr(block, "text", ""))

                return " ".join(extracted_text)

        return ""

    async def fork_process(self) -> "LLMProcess":
        """Create a deep copy of this process with preserved state.

        This implements the fork system call semantics where a copy of the
        process is created with the same state and configuration. The forked
        process is completely independent and can run separate tasks.

        Returns:
            A new LLMProcess instance that is a deep copy of this one
        """
        # Create a new instance of LLMProcess with the same program
        forked_process = LLMProcess(program=self.program)

        # Copy the enriched system prompt if it exists
        if hasattr(self, "enriched_system_prompt") and self.enriched_system_prompt:
            forked_process.enriched_system_prompt = self.enriched_system_prompt

        # Deep copy the conversation state
        forked_process.state = copy.deepcopy(self.state)

        # Copy any preloaded content
        if hasattr(self, "preloaded_content") and self.preloaded_content:
            forked_process.preloaded_content = copy.deepcopy(self.preloaded_content)

        # No need to copy tools and tool_handlers - they are properties now

        # If the parent process had MCP enabled, set it in the fork
        if self.mcp_enabled:
            forked_process.mcp_enabled = True
            
            # Copy the MCPManager reference from tool_manager if it exists
            if hasattr(self.tool_manager, "mcp_manager") and self.tool_manager.mcp_manager:
                # We don't need to create a new mcp_manager, just share the reference
                # since tool_manager already handles initialization
                pass

        # If the parent process had file descriptors enabled, copy the manager and its state
        if self.file_descriptor_enabled and self.fd_manager:
            forked_process.file_descriptor_enabled = True
            forked_process.fd_manager = copy.deepcopy(self.fd_manager)

            # Copy references_enabled setting
            forked_process.references_enabled = getattr(self, "references_enabled", False)

            # Ensure user input handling settings are copied correctly
            if not hasattr(forked_process.fd_manager, "page_user_input"):
                forked_process.fd_manager.page_user_input = getattr(self.fd_manager, "page_user_input", False)
            if not hasattr(forked_process.fd_manager, "max_input_chars"):
                forked_process.fd_manager.max_input_chars = getattr(self.fd_manager, "max_input_chars", 8000)

        # Prevent forked processes from forking again
        forked_process.allow_fork = False

        # Preserve any other state we need
        # Note: We don't copy tool handlers as they're already set up in the constructor

        return forked_process
