"""LLMProcess class for executing LLM programs and handling interactions."""

import asyncio
import copy
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from llmproc.common.results import RunResult, ToolResult
from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.program import LLMProgram
from llmproc.providers import get_provider_client
from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor
from llmproc.providers.constants import ANTHROPIC_PROVIDERS, GEMINI_PROVIDERS
from llmproc.providers.gemini_process_executor import GeminiProcessExecutor
from llmproc.providers.openai_process_executor import OpenAIProcessExecutor
from llmproc.tools import ToolManager, file_descriptor_instructions

load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)


class LLMProcess:
    """Process for interacting with LLMs using standardized program definitions."""

    def __init__(
        self,
        # Program reference
        program: LLMProgram,
        # Model and provider information
        model_name: str,
        provider: str,
        original_system_prompt: str,
        system_prompt: str,
        display_name: Optional[str] = None,
        base_dir: Optional[Path] = None,
        api_params: dict[str, Any] = None,
        # Runtime state
        state: list[dict[str, Any]] = None,
        enriched_system_prompt: Optional[str] = None,
        allow_fork: bool = True,
        # Client
        client: Any = None,
        # File descriptor system
        fd_manager: Optional[FileDescriptorManager] = None,
        file_descriptor_enabled: bool = False,
        references_enabled: bool = False,
        # Linked programs
        linked_programs: dict[str, LLMProgram] = None,
        linked_program_descriptions: dict[str, str] = None,
        has_linked_programs: Optional[bool] = None,
        # Tool management
        tool_manager: ToolManager = None,
        enabled_tools: list[str] = None,
        # MCP configuration
        mcp_config_path: Optional[str] = None,
        mcp_tools: dict[str, Any] = None,
        mcp_enabled: Optional[bool] = None,
    ) -> None:
        """Initialize LLMProcess with pre-computed state.

        ⚠️ WARNING: DO NOT USE THIS CONSTRUCTOR DIRECTLY! ⚠️

        ALWAYS use the async factory method `await program.start()` instead,
        which properly handles initialization following the Unix-inspired pattern:

        ```python
        program = LLMProgram.from_toml("config.toml")
        process = await program.start()  # CORRECT WAY TO CREATE PROCESS
        ```

        This constructor expects pre-computed state from program_exec.prepare_process_state
        and is not designed for direct use.

        Args:
            program: The compiled LLMProgram reference
            model_name: Model name
            provider: Provider name
            original_system_prompt: Original system prompt
            system_prompt: System prompt (may be modified)
            display_name: Display name
            base_dir: Base directory
            api_params: API parameters
            state: Conversation state
            enriched_system_prompt: Enriched system prompt
            allow_fork: Whether forking is allowed
            client: Provider client
            fd_manager: File descriptor manager
            file_descriptor_enabled: Whether file descriptor system is enabled
            references_enabled: Whether references are enabled
            linked_programs: Linked programs
            linked_program_descriptions: Linked program descriptions
            has_linked_programs: Whether linked programs exist
            tool_manager: Tool manager
            enabled_tools: Enabled tools
            mcp_config_path: MCP config path
            mcp_tools: MCP tools
            mcp_enabled: Whether MCP is enabled
            _tools_need_initialization: Whether tools need initialization

        Raises:
            ValueError: If required parameters are missing
        """
        # Basic validation for required parameters
        if not model_name or not provider:
            raise ValueError("model_name and provider are required for LLMProcess initialization")

        # Store all provided state attributes
        self.program = program
        self.model_name = model_name
        self.provider = provider
        self.original_system_prompt = original_system_prompt
        self.system_prompt = system_prompt
        self.display_name = display_name
        self.base_dir = base_dir
        self.api_params = api_params or {}
        self.parameters = {}  # Parameters are already processed in program

        # Runtime state
        self.state = state or []
        self.enriched_system_prompt = enriched_system_prompt
        self.allow_fork = allow_fork

        # Client
        self.client = client

        # File descriptor system
        self.fd_manager = fd_manager
        self.file_descriptor_enabled = file_descriptor_enabled
        self.references_enabled = references_enabled

        # Linked programs
        self.linked_programs = linked_programs or {}
        self.linked_program_descriptions = linked_program_descriptions or {}
        self.has_linked_programs = has_linked_programs if has_linked_programs is not None else bool(linked_programs)

        # Tool management
        self.tool_manager = tool_manager
        self.enabled_tools = enabled_tools or []

        # MCP configuration
        self.mcp_config_path = mcp_config_path
        self.mcp_tools = mcp_tools or {}
        self.mcp_enabled = mcp_enabled if mcp_enabled is not None else (mcp_config_path is not None)

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

        # MCP tools should already be initialized during program.start()
        # No need for lazy initialization here

        # Process user input through file descriptor manager if enabled
        processed_user_input = user_input
        if self.file_descriptor_enabled:  # fd_manager is guaranteed to exist if enabled flag is true
            # Delegate to file descriptor manager to handle large user input
            processed_user_input = self.fd_manager.handle_user_input(user_input)

            # Log if input was converted to a file descriptor
            if processed_user_input != user_input:
                logger.info(f"Large user input ({len(user_input)} chars) converted to file descriptor")

        # NOTE: We don't add the user message to the state here.
        # Provider-specific executors will add it with proper IDs and formatting.
        # This prevents duplicate user messages in the conversation history.

        # Create provider-specific process executors
        if self.provider == "openai":
            # Use the OpenAI process executor (simplified version)
            executor = OpenAIProcessExecutor()
            run_result = await executor.run(self, processed_user_input, max_iterations, callbacks, run_result)

        elif self.provider in ANTHROPIC_PROVIDERS:
            # Use the stateless AnthropicProcessExecutor for both direct Anthropic API and Vertex AI
            executor = AnthropicProcessExecutor()
            run_result = await executor.run(self, processed_user_input, max_iterations, callbacks, run_result)

        elif self.provider in GEMINI_PROVIDERS:
            # Use the GeminiProcessExecutor for both direct API and Vertex AI
            executor = GeminiProcessExecutor()
            run_result = await executor.run(self, processed_user_input, max_iterations, callbacks, run_result)
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

    def reset_state(self, keep_system_prompt: bool = True, keep_file_descriptors: bool = True) -> None:
        """Reset the conversation state.

        Args:
            keep_system_prompt: Whether to keep the system prompt for the next API call
            keep_file_descriptors: Whether to keep file descriptor content

        Note:
            State only contains user/assistant messages, not system message.
            System message is stored separately in enriched_system_prompt.
            Preloaded content is now considered immutable and is always kept.
        """
        # Clear the conversation state (user/assistant messages)
        self.state = []

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

        This method follows the standard Unix pattern where:
        1. A new base process is created through the standard program_exec.create_process path
        2. Runtime state is copied from parent to child
        3. State-specific configurations are preserved

        Returns:
            A new LLMProcess instance that is a deep copy of this one

        Raises:
            RuntimeError: If forking is not allowed for this process
        """
        # Check if forking is allowed for this process
        if not self.allow_fork:
            raise RuntimeError("Forking is not allowed for this process")

        # Get a display name for logging, using fallbacks for test environments
        display_name = "unknown"
        try:
            if hasattr(self.program, "display_name") and self.program.display_name:
                display_name = self.program.display_name
            elif hasattr(self, "model_name") and self.model_name:
                display_name = self.model_name
        except (AttributeError, TypeError):
            # Handle mock objects or missing attributes in tests
            pass

        logger.info(f"Forking process for program: {display_name}")

        # 1. Create a new base process using the standard program_exec.create_process flow
        # This ensures all dependencies are properly initialized through the same path
        # as program.start(), maintaining consistency across the codebase
        from llmproc.program_exec import create_process

        forked_process = await create_process(self.program)

        # 2. Copy runtime state from parent to child
        # Deep copy to ensure complete independence between parent and child processes
        forked_process.state = copy.deepcopy(self.state)

        # Copy the enriched system prompt if it exists
        if hasattr(self, "enriched_system_prompt") and self.enriched_system_prompt:
            forked_process.enriched_system_prompt = self.enriched_system_prompt

        # 3. Deep copy FD manager state if FD is enabled
        # File descriptor system state needs special handling to ensure
        # complete independence between parent and child
        if (
            hasattr(self, "file_descriptor_enabled")
            and self.file_descriptor_enabled
            and hasattr(self, "fd_manager")
            and self.fd_manager
        ):
            forked_process.file_descriptor_enabled = True
            forked_process.fd_manager = copy.deepcopy(self.fd_manager)

            # Copy references_enabled setting
            forked_process.references_enabled = getattr(self, "references_enabled", False)

            # Ensure user input handling settings are copied correctly
            if hasattr(self.fd_manager, "file_descriptors") and hasattr(forked_process, "fd_manager"):
                if not hasattr(forked_process.fd_manager, "page_user_input"):
                    forked_process.fd_manager.page_user_input = getattr(self.fd_manager, "page_user_input", False)
                if not hasattr(forked_process.fd_manager, "max_input_chars"):
                    forked_process.fd_manager.max_input_chars = getattr(self.fd_manager, "max_input_chars", 8000)

                try:
                    fd_count = len(getattr(forked_process.fd_manager, "file_descriptors", {}))
                    logger.debug(f"Deep copied FD manager state during fork. Copied {fd_count} FDs.")
                except (AttributeError, TypeError):
                    # Handle mock objects in tests
                    pass

        # 4. Prevent the forked process from forking again
        # This prevents potential issues with deep fork chains
        forked_process.allow_fork = False

        # Get forked display name for logging, using fallbacks for test environments
        forked_display = "unknown"
        try:
            if hasattr(forked_process, "display_name") and forked_process.display_name:
                forked_display = forked_process.display_name
            elif hasattr(forked_process, "model_name") and forked_process.model_name:
                forked_display = forked_process.model_name
        except (AttributeError, TypeError):
            # Handle mock objects or missing attributes in tests
            pass

        logger.info(f"Fork successful. New process created for {forked_display}")
        return forked_process
