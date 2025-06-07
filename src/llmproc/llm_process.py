"""LLMProcess class for executing LLM programs and handling interactions."""

import asyncio
import inspect
import logging
import threading
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from llmproc.callbacks import CallbackEvent
from llmproc.callbacks.callback_utils import (
    add_callback as utils_add_callback,
)
from llmproc.callbacks.callback_utils import (
    trigger_event as utils_trigger_event,
)
from llmproc.common.access_control import AccessLevel
from llmproc.common.results import RunResult, ToolResult
from llmproc.event_loop_mixin import EventLoopMixin
from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.process_forking import ProcessForkingMixin
from llmproc.program import LLMProgram
from llmproc.providers.utils import choose_provider_executor
from llmproc.tools import ToolManager

load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)


class LLMProcess(EventLoopMixin, ProcessForkingMixin):
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
        access_level: AccessLevel = AccessLevel.ADMIN,
        display_name: Optional[str] = None,
        base_dir: Optional[Path] = None,
        api_params: dict[str, Any] = None,
        # Runtime state
        state: list[dict[str, Any]] = None,
        enriched_system_prompt: Optional[str] = None,
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
        mcp_servers: dict[str, Any] | None = None,
        mcp_tools: dict[str, Any] = None,
        mcp_enabled: Optional[bool] = None,
        # User prompt configuration
        user_prompt: Optional[str] = None,
        max_iterations: int = 10,
        # Internal event loop (injected by program_exec in async path)
        loop: asyncio.AbstractEventLoop | None = None,
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
            mcp_servers: Embedded MCP servers dictionary
            mcp_tools: MCP tools
            mcp_enabled: Whether MCP is enabled
            access_level: Access level for tool execution
            user_prompt: Custom user prompt for the process
            max_iterations: Maximum tool iterations per run
            loop: Optional event loop to use for asynchronous execution

        Raises:
            ValueError: If required parameters are missing
        """
        # Basic validation for required parameters
        if not model_name or not provider:
            raise ValueError("model_name and provider are required for LLMProcess initialization")

        # Initialize event loop handling
        EventLoopMixin.__init__(self, loop)

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

        # Client
        self.client = client

        # Initialize provider-specific executor
        self.executor = choose_provider_executor(provider)

        # File descriptor system
        self.fd_manager = fd_manager
        self.file_descriptor_enabled = file_descriptor_enabled
        self.references_enabled = references_enabled

        # Linked programs
        self.linked_programs = linked_programs or {}
        self.linked_program_descriptions = linked_program_descriptions or {}
        self.has_linked_programs = has_linked_programs if has_linked_programs is not None else bool(linked_programs)

        # Tool management and access control
        self.tool_manager = tool_manager
        self.enabled_tools = enabled_tools or []
        self.access_level = access_level
        if tool_manager:
            self.tool_manager.set_process_access_level(access_level)

        # MCP configuration
        self.mcp_config_path = mcp_config_path
        self.mcp_servers = mcp_servers
        self.mcp_tools = mcp_tools or {}
        self.mcp_enabled = (
            mcp_enabled if mcp_enabled is not None else (mcp_config_path is not None or mcp_servers is not None)
        )

        # User prompt configuration
        self.user_prompt = user_prompt
        self.max_iterations = max_iterations

        # Buffer for stderr log entries
        self.stderr_log: list[str] = []

        # Callback system - initialize empty list
        self.callbacks = []

    def add_callback(self, callback: Callable) -> "LLMProcess":
        """Register a callback function or object."""
        utils_add_callback(self, callback)
        return self

    # ------------------------------------------------------------------
    # Private event-loop helpers
    # ------------------------------------------------------------------

    def trigger_event(self, event: CallbackEvent, *args, **kwargs) -> None:
        """Trigger an event to all registered callbacks."""
        utils_trigger_event(self, event, *args, **kwargs)

    def _process_user_input(self, user_input: str) -> str:
        """Process user input through the file descriptor system if enabled."""
        if not user_input or user_input.strip() == "":
            raise ValueError("User input cannot be empty")

        processed = user_input
        if self.file_descriptor_enabled:
            processed = self.fd_manager.handle_user_input(user_input)
            if processed != user_input:
                logger.info(f"Large user input ({len(user_input)} chars) converted to file descriptor")
        return processed

    def _process_response_references(self) -> None:
        """Process reference IDs in the last assistant message."""
        if not self.file_descriptor_enabled:
            return

        if self.state and self.state[-1].get("role") == "assistant":
            assistant_message = self.state[-1].get("content", "")

            if isinstance(assistant_message, list):
                for block in assistant_message:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content = block.get("text", "")
                        self.fd_manager.process_references(text_content)
            else:
                self.fd_manager.process_references(assistant_message)

    async def run(self, user_input: str, max_iterations: int = None) -> "RunResult":
        """Run the LLM process with user input asynchronously.

        This method must be called from an async context with 'await'.
        For synchronous execution, use run_sync() instead.

        Args:
            user_input: The user message to process
            max_iterations: Maximum number of tool-calling iterations

        Returns:
            RunResult object with execution metrics
        """
        # Use default max_iterations if not specified
        if max_iterations is None:
            max_iterations = self.max_iterations

        # Get the current running loop
        current_loop = asyncio.get_running_loop()

        # Check if we're in our process loop
        if current_loop is self._loop:
            # Direct execution in our loop
            return await self._async_run(user_input, max_iterations)
        else:
            # Running in a different loop - bridge to our loop
            future = self._submit_to_loop(self._async_run(user_input, max_iterations))
            return await asyncio.wrap_future(future)

    async def _async_run(self, user_input: str, max_iterations: int) -> "RunResult":
        """Internal async implementation of run.

        Args:
            user_input: The user message to process
            max_iterations: Maximum number of tool-calling iterations

        Returns:
            RunResult object with execution metrics

        Raises:
            ValueError: If user_input is empty
        """
        # Create a RunResult object to track this run
        run_result = RunResult()

        processed_user_input = self._process_user_input(user_input)
        run_result = await self.executor.run(self, processed_user_input, max_iterations, run_result)
        self._process_response_references()

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

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    async def aclose(self, timeout: float = 2.0) -> None:
        """Asynchronously close background resources and stop the loop.

        Args:
            timeout: Maximum time in seconds to wait for cleanup (default: 2.0)
        """
        # Attempt to close MCP connections gracefully with timeout
        try:
            mcp_manager = getattr(self.tool_manager, "mcp_manager", None)
            aggregator = getattr(mcp_manager, "aggregator", None) if mcp_manager else None
            if aggregator is not None:
                # Use asyncio.wait_for to apply timeout for MCP client closing
                await asyncio.wait_for(aggregator.close_clients(), timeout=timeout / 2)
        except TimeoutError:
            logger.warning(f"Timeout while closing MCP clients after {timeout / 2} seconds")
        except Exception as exc:  # noqa: BLE001 – best-effort
            logger.warning("Error while closing MCP clients: %s", exc)

        # Stop private loop if we own it
        if self._own_loop and self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=timeout)

    async def call_tool(self, tool_name: str, args_dict: dict[str, Any]) -> ToolResult:
        """Call a tool by name with the given arguments dictionary.

        Internal helper used by ``LLMProgram.start_sync`` and the async API.

        This delegates to :class:`ToolManager` for execution.

        Args:
            tool_name: Name of the tool to call.
            args_dict: Arguments dictionary for the tool.

        Returns:
            The result from the tool or an error ``ToolResult`` instance.
        """
        try:
            return await self.tool_manager.call_tool(tool_name, args_dict)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"Error calling tool '{tool_name}': {exc}"
            logger.error(error_msg, exc_info=True)
            return ToolResult.from_error(error_msg)

    async def count_tokens(self):
        """Count tokens in the current conversation state using the executor.

        Returns:
            dict | None: Token count information with provider-specific details
            or ``None`` if token counting is unsupported. The returned
            dictionary may include:
                - ``input_tokens``: Number of tokens in conversation
                - ``context_window``: Max tokens supported by the model
                - ``percentage``: Percentage of context window used
                - ``remaining_tokens``: Number of tokens left in context window
                - ``cached_tokens``: (Gemini only) Number of tokens in cached content
                - ``note``: Informational message when estimation is used
                - ``error``: Error message if token counting failed
        """
        # Use the provider-specific executor configured for this process. The
        # executor map is used during initialization, so ``self.executor`` is
        # responsible for implementing ``count_tokens`` when supported.
        if hasattr(self.executor, "count_tokens"):
            return await self.executor.count_tokens(self)

        # No token counting support for this executor
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

    def get_stderr_log(self) -> list[str]:
        """Return the accumulated stderr log entries."""
        return self.stderr_log.copy()

    @property
    def run_stop_reason(self) -> str | None:
        """Deprecated: Use RunResult.stop_reason instead.

        This property provides backward compatibility but will be removed in a future version.
        Stop reasons are now tracked per-run in RunResult objects rather than on the process.

        Returns:
            None (stop reasons are no longer stored on the process)
        """
        warnings.warn(
            "process.run_stop_reason is deprecated. Use run_result.stop_reason instead. "
            "Stop reasons are now tracked per-run in RunResult objects.",
            DeprecationWarning,
            stacklevel=2,
        )
        return None

    @run_stop_reason.setter
    def run_stop_reason(self, value: str | None) -> None:
        """Deprecated: Use RunResult.set_stop_reason() instead."""
        warnings.warn(
            "Setting process.run_stop_reason is deprecated. Use run_result.set_stop_reason() instead. "
            "Stop reasons are now tracked per-run in RunResult objects.",
            DeprecationWarning,
            stacklevel=2,
        )


# Create AsyncLLMProcess alias for more explicit API usage
class AsyncLLMProcess(LLMProcess):
    """Alias for LLMProcess with explicit naming for async usage.

    This class is functionally identical to LLMProcess, but provides a clearer
    naming convention for code that specifically wants to use the async API.

    Example:
        ```python
        # Both are identical in functionality:
        process1 = await program.start()  # Returns LLMProcess
        process2 = await program.start()  # Can be treated as AsyncLLMProcess
        assert isinstance(process2, AsyncLLMProcess)  # True
        ```
    """

    pass


# Synchronous wrapper for LLMProcess
class SyncLLMProcess(LLMProcess):
    """Synchronous wrapper for LLMProcess with blocking methods.

    This class inherits from LLMProcess but provides synchronous versions of
    all async public API methods, making it easier to use in synchronous contexts.

    Example:
        ```python
        # Synchronous usage
        process = program.start_sync()  # Returns SyncLLMProcess
        result = process.run("Hello")   # Blocking call
        process.close()                 # Blocking cleanup
        ```

    Internal async methods like _fork_process are inherited from LLMProcess
    and remain async. They are intended for internal tool use only.
    """

    def __init__(self, _loop: Optional[asyncio.AbstractEventLoop] = None, **kwargs):
        """Initialize with parameters from program_exec.create_sync_process.

        ⚠️ WARNING: DO NOT USE THIS CONSTRUCTOR DIRECTLY! ⚠️

        ALWAYS use the synchronous factory method `program.start_sync()` instead,
        which properly handles initialization following the Unix-inspired pattern:

        ```python
        program = LLMProgram.from_toml("config.toml")
        process = program.start_sync()  # CORRECT WAY TO CREATE SYNCLLMPROCESS
        ```

        Args:
            _loop: Optional event loop to use (creates a new one if not provided)
            **kwargs: All parameters passed to parent LLMProcess
        """
        super().__init__(**kwargs)

        # Create a new event loop if one wasn't provided
        self._loop = _loop or asyncio.new_event_loop()
        self._own_loop = _loop is None  # We own the loop if we created it

    # Synchronous wrappers for async methods (public API only)

    def run(self, user_input: str, max_iterations: Optional[int] = None) -> RunResult:
        """Run the LLM process with user input synchronously.

        Args:
            user_input: The user message to process
            max_iterations: Maximum number of tool-calling iterations

        Returns:
            RunResult object with execution metrics
        """
        logger.debug(f"Running SyncLLMProcess with input: {user_input[:50]}...")
        return self._loop.run_until_complete(super().run(user_input, max_iterations))

    def count_tokens(self) -> dict[str, Any]:
        """Count tokens in the conversation synchronously.

        Returns:
            Token count information with provider-specific details
        """
        return self._loop.run_until_complete(super().count_tokens())

    def close(self) -> None:
        """Clean up resources synchronously.

        This method performs all the async cleanup steps synchronously and
        also closes the event loop if this process owns it.
        """
        try:
            # Run the async cleanup
            self._loop.run_until_complete(super().aclose())
        finally:
            # Close our loop if we own it
            if self._own_loop and not self._loop.is_closed():
                self._loop.close()
