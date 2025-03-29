"""LLMProcess class for executing LLM programs and handling interactions."""

import asyncio
import logging
import os
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llmproc.program import LLMProgram
from llmproc.providers import get_provider_client
from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor
from llmproc.providers.openai_process_executor import OpenAIProcessExecutor
from llmproc.results import RunResult
from llmproc.tools import ToolRegistry, file_descriptor_instructions, mcp, register_system_tools
from llmproc.tools.file_descriptor import FileDescriptorManager

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
            For async initialization (MCP), use the create() factory method instead.
        """
        # Store the program reference
        self.program = program

        # Extract core attributes from program
        self.model_name = program.model_name
        self.provider = program.provider
        self.system_prompt = (
            program.system_prompt
        )  # Basic system prompt without enhancements
        self.display_name = program.display_name
        self.base_dir = program.base_dir
        self.api_params = program.api_params
        self.parameters = {}  # Keep empty - parameters are already processed in program

        # Initialize state for preloaded content
        self.preloaded_content = {}

        # Track the enriched system prompt (will be set on first run)
        self.enriched_system_prompt = None

        # Extract tool configuration
        self.enabled_tools = []
        if hasattr(program, "tools") and program.tools:
            # Get enabled tools from the program
            self.enabled_tools = program.tools.get("enabled", [])

        # Initialize the tool registry
        self.tool_registry = ToolRegistry()

        # MCP Configuration
        self.mcp_enabled = False
        self.mcp_config_path = getattr(program, "mcp_config_path", None)
        self.mcp_tools = getattr(program, "mcp_tools", {})
        self._mcp_initialized = False

        # Mark if we need async initialization
        self._needs_async_init = self.mcp_config_path is not None and bool(
            self.mcp_tools
        )

        # Linked Programs Configuration
        self.linked_programs = {}
        self.has_linked_programs = False
        self.linked_program_descriptions = {}

        # Initialize linked programs if provided in constructor
        if linked_programs_instances:
            self.has_linked_programs = True
            self.linked_programs = linked_programs_instances
        # Otherwise use linked programs from the program - store as program objects
        elif hasattr(program, "linked_programs") and program.linked_programs:
            self.has_linked_programs = True
            self.linked_programs = program.linked_programs
            
        # Get linked program descriptions if available
        if hasattr(program, "linked_program_descriptions") and program.linked_program_descriptions:
            self.linked_program_descriptions = program.linked_program_descriptions

        # Initialize system tools
        self._initialize_tools()

        # Get project_id and region for Vertex if provided in parameters
        project_id = getattr(program, "project_id", None)
        region = getattr(program, "region", None)

        # Check if OpenAI provider is used with tools (currently not supported)
        if self.provider == "openai" and self.enabled_tools:
            raise ValueError(
                "Tool usage is not yet supported for OpenAI models in this implementation. "
                "Please use a model without tools, or use the Anthropic provider for tool support."
            )

        # Initialize the client
        self.client = get_provider_client(
            self.provider, self.model_name, project_id, region
        )

        # Store the original system prompt before any files are preloaded
        self.original_system_prompt = self.system_prompt

        # Initialize conversation state
        # Note: State contains only user/assistant messages, not system message
        # System message is kept separately and passed directly to the API
        self.state = []

        # Initialize fork support
        self.allow_fork = True  # By default, allow forking
        
        # Initialize file descriptor system
        self.file_descriptor_enabled = False
        self.fd_manager = None
        self.references_enabled = False
        
        # Check for file descriptor configuration in program
        if hasattr(program, "file_descriptor"):
            # Configure file descriptor manager with program settings
            fd_config = program.file_descriptor
            
            # Enable if explicitly enabled in config or if read_fd is in enabled tools
            explicit_enabled = fd_config.get("enabled", False)
            implicit_enabled = "read_fd" in self.enabled_tools
            
            if explicit_enabled or implicit_enabled:
                self.file_descriptor_enabled = True
                
                # Get configuration values with defaults - fd_config is a dictionary, not an object
                default_page_size = fd_config.get("default_page_size", 4000)
                max_direct_output_chars = fd_config.get("max_direct_output_chars", 8000)
                max_input_chars = fd_config.get("max_input_chars", 8000)
                page_user_input = fd_config.get("page_user_input", True)
                
                # Check if references are enabled - fd_config is a dictionary, not an object
                self.references_enabled = fd_config.get("enable_references", False)
                
                # Initialize the file descriptor manager
                self.fd_manager = FileDescriptorManager(
                    default_page_size=default_page_size,
                    max_direct_output_chars=max_direct_output_chars,
                    max_input_chars=max_input_chars,
                    page_user_input=page_user_input
                )
                
                logger.info(
                    f"File descriptor system enabled with page size {default_page_size}, "
                    f"user input paging: {page_user_input}, "
                    f"references: {self.references_enabled}"
                )
        
        # If read_fd is in tools but no configuration provided, still enable with defaults
        elif "read_fd" in self.enabled_tools:
            self.file_descriptor_enabled = True
            self.fd_manager = FileDescriptorManager()
            logger.info("File descriptor system enabled with default settings")

        # Preload files if specified
        if hasattr(program, "preload_files") and program.preload_files:
            self.preload_files(program.preload_files)

    @classmethod
    async def create(
        cls,
        program: LLMProgram,
        linked_programs_instances: dict[str, "LLMProcess"] | None = None,
    ) -> "LLMProcess":
        """Create and fully initialize an LLMProcess asynchronously.

        This factory method handles async initialization in a clean way,
        ensuring the instance is fully ready to use when returned.

        Args:
            program: The LLMProgram to use
            linked_programs_instances: Dictionary of pre-initialized LLMProcess instances

        Returns:
            A fully initialized LLMProcess

        Raises:
            All exceptions from __init__, plus:
            RuntimeError: If MCP initialization fails
        """
        # Create instance with basic initialization
        instance = cls(program, linked_programs_instances)

        # Perform async initialization if needed
        if instance._needs_async_init:
            instance.mcp_enabled = True
            await instance._initialize_mcp_tools()

        return instance

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


    async def run(
        self, user_input: str, max_iterations: int = 10, callbacks: dict = None
    ) -> "RunResult":
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

    async def _async_run(
        self, user_input: str, max_iterations: int = 10, callbacks: dict = None
    ) -> "RunResult":
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
        # Import the RunResult class
        from llmproc.results import RunResult

        # Create a RunResult object to track this run
        run_result = RunResult()

        # Normalize callbacks
        callbacks = callbacks or {}

        # Verify user input isn't empty
        if not user_input or user_input.strip() == "":
            raise ValueError("User input cannot be empty")

        # MCP tools should already be initialized during program.start()
        # No need for lazy initialization here

        # Generate enriched system prompt on first run
        if self.enriched_system_prompt is None:
            self.enriched_system_prompt = self.program.get_enriched_system_prompt(
                process_instance=self, include_env=True
            )
            
        # Process user input through file descriptor manager if enabled
        processed_user_input = user_input
        if self.file_descriptor_enabled and self.fd_manager:
            # Handle large user input (convert to file descriptor if needed)
            processed_user_input = self.fd_manager.handle_user_input(user_input)
            
            # Log if input was converted to a file descriptor
            if processed_user_input != user_input:
                logger.info(
                    f"Large user input ({len(user_input)} chars) converted to file descriptor"
                )

        # Add processed user input to state
        self.state.append({"role": "user", "content": processed_user_input})

        # Create provider-specific process executors
        if self.provider == "openai":
            # Use the OpenAI process executor (simplified version)
            executor = OpenAIProcessExecutor()
            run_result = await executor.run(
                self, user_input, max_iterations, callbacks, run_result
            )

        elif self.provider in ["anthropic", "anthropic_vertex"]:
            # Use the stateless AnthropicProcessExecutor for both direct Anthropic API and Vertex AI
            executor = AnthropicProcessExecutor()
            run_result = await executor.run(
                self, user_input, max_iterations, callbacks, run_result
            )
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")

        # Process any references in the last assistant message for reference ID system
        if self.file_descriptor_enabled and self.fd_manager and self.references_enabled:
            # Get the last assistant message if available
            if self.state and len(self.state) > 0 and self.state[-1].get("role") == "assistant":
                assistant_message = self.state[-1].get("content", "")
                
                # Check if we have a string message or a structured message (Anthropic)
                if isinstance(assistant_message, list):
                    # Process each text block in the message
                    for i, block in enumerate(assistant_message):
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content = block.get("text", "")
                            references = self.fd_manager.extract_references(text_content)
                            if references:
                                logger.info(f"Extracted {len(references)} references from assistant message")
                else:
                    # Process the simple string message
                    references = self.fd_manager.extract_references(assistant_message)
                    if references:
                        logger.info(f"Extracted {len(references)} references from assistant message")

        # Mark the run as complete and calculate duration
        run_result.complete()

        return run_result

    def get_state(self) -> list[dict[str, str]]:
        """Return the current conversation state.

        Returns:
            A copy of the current conversation state
        """
        return self.state.copy()

    async def _initialize_mcp_tools(self) -> None:
        """Initialize MCP registry and tools.

        This sets up the MCP registry and filters tools based on user configuration.
        Only tools explicitly specified in the mcp_tools configuration will be enabled.
        Only servers that have tools configured will be initialized.
        """
        if not self.mcp_enabled:
            return

        success = await mcp.initialize_mcp_tools(
            self, self.tool_registry, self.mcp_config_path, self.mcp_tools
        )

        if not success:
            logger.warning(
                "Failed to initialize MCP tools. Some features may not work correctly."
            )

    def reset_state(
        self, keep_system_prompt: bool = True, keep_preloaded: bool = True,
        keep_file_descriptors: bool = True
    ) -> None:
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
                page_user_input=self.fd_manager.page_user_input
            )
            # Copy over the FD-related tools registry
            self.fd_manager.fd_related_tools = self.fd_manager.fd_related_tools.union(
                self.fd_manager._FD_RELATED_TOOLS
            )

        # Always reset the enriched system prompt - it will be regenerated on next run
        # with the correct combination of system prompt and preloaded content
        self.enriched_system_prompt = None

    def _initialize_tools(self) -> None:
        """Initialize all system tools.

        MCP tools require async initialization and are handled separately
        via the create() factory method or lazy initialization in run().
        """
        # Validate MCP configuration if present
        if self._needs_async_init:
            if not HAS_MCP:
                raise ImportError(
                    "MCP features require the mcp-registry package. Install it with 'pip install mcp-registry'."
                )

            # Currently only support Anthropic with MCP
            if self.provider != "anthropic":
                raise ValueError(
                    "MCP features are currently only supported with the Anthropic provider"
                )

        # Register system tools using the ToolRegistry
        register_system_tools(self.tool_registry, self)

    @property
    def tools(self) -> list:
        """Property to access tool definitions.

        Returns:
            List of tool definitions.
        """
        return self.tool_registry.get_definitions()

    @property
    def tool_handlers(self) -> dict:
        """Property to access tool handlers.

        Returns:
            Dictionary of tool handlers.
        """
        return self.tool_registry.tool_handlers

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
        try:
            return await self.tool_registry.call_tool(tool_name, args)
        except Exception as e:
            raise RuntimeError(f"Error executing tool '{tool_name}': {str(e)}")

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
        import copy

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

        # If the parent process had MCP initialized, replicate the state in the fork
        if self.mcp_enabled and self._mcp_initialized:
            forked_process.mcp_enabled = True
            forked_process._mcp_initialized = True

            # Copy the aggregator if it exists
            if hasattr(self, "aggregator"):
                forked_process.aggregator = self.aggregator
                
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
