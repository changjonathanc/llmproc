"""LLMProcess class for executing LLM programs and handling interactions."""

import asyncio
import copy
import inspect
import logging
import threading
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from llmproc.callbacks import CallbackEvent
from llmproc.common.access_control import AccessLevel
from llmproc.common.results import RunResult, ToolResult
from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.process_snapshot import ProcessSnapshot
from llmproc.program import LLMProgram
from llmproc.providers import EXECUTOR_MAP
from llmproc.providers.utils import choose_provider_executor
from llmproc.tools import ToolManager

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

        # --------------------------------------------------------------
        # Event-loop bookkeeping – store loop reference supplied by
        # program_exec (async path) or leave as *None* (will be created on
        # first synchronous call).
        # --------------------------------------------------------------

        self._loop: asyncio.AbstractEventLoop | None = loop
        self._loop_thread: threading.Thread | None = None
        self._own_loop: bool = False

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
        executor_cls = EXECUTOR_MAP.get(provider)
        if executor_cls is None:
            logger.warning(
                "Unknown provider '%s' requested; falling back to default executor",
                provider,
            )
            self.executor = choose_provider_executor(self)
        else:
            self.executor = executor_cls()

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

        # Thread that drives the private event loop when one is created via
        # _ensure_loop_thread().  This is *not* started when an external loop
        # is supplied.
        self._loop_thread: threading.Thread | None = None
        self._own_loop: bool = False

    def add_callback(self, callback: Callable) -> "LLMProcess":
        """Register a callback function or object.

        You can pass either:
        1. A function that accepts ``(event, *args, **kwargs)``
        2. An object with methods matching event names (``tool_start``,
           ``tool_end``, ``response``)

        Callbacks may be ``async`` or regular functions. ``async`` callbacks are
        automatically scheduled on the process event loop. Callback objects may
        freely mix synchronous and asynchronous methods.

        Args:
            callback: The callback function or object

        Returns:
            Self for method chaining
        """
        self.callbacks.append(callback)
        return self

    # ------------------------------------------------------------------
    # Private event-loop helpers
    # ------------------------------------------------------------------

    def _ensure_loop_thread(self) -> None:
        """Create and start a dedicated event loop thread if necessary."""
        if self._loop is not None:
            return

        self._loop = asyncio.new_event_loop()
        self._own_loop = True

        def _runner() -> None:  # runs in background thread
            asyncio.set_event_loop(self._loop)  # type: ignore[arg-type]
            self._loop.run_forever()

        # Use unique process identifier in thread name for easier debugging
        thread_name = f"LLMProcessLoop-{id(self):x}"
        self._loop_thread = threading.Thread(target=_runner, name=thread_name, daemon=True)
        self._loop_thread.start()

    def _submit_to_loop(self, coro):  # type: ignore[valid-type]
        """Schedule coroutine *coro* on the process loop and return a Future."""
        self._ensure_loop_thread()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def trigger_event(self, event: CallbackEvent, *args, **kwargs) -> None:
        """Trigger an event to all registered callbacks.

        Args:
            event: The CallbackEvent enum value
            *args: Positional arguments to pass to the callback
            **kwargs: Keyword arguments to pass to the callback
        """
        if not self.callbacks:
            return

        event_name = event.value  # Get string value from enum

        for callback in self.callbacks:
            try:
                method = None
                pass_event = False

                # Class-based callback with method matching event name
                if hasattr(callback, event_name):
                    method = getattr(callback, event_name)
                elif callable(callback):
                    method = callback
                    pass_event = True

                if method is None:
                    continue

                if inspect.iscoroutinefunction(method):
                    if pass_event:
                        fut = self._submit_to_loop(method(event, *args, **kwargs))
                    else:
                        fut = self._submit_to_loop(method(*args, **kwargs))
                    fut.add_done_callback(
                        lambda f, en=event.name: logger.warning(f"Error in {en} callback: {f.exception()}")
                        if f.exception()
                        else None
                    )
                else:
                    if pass_event:
                        result = method(event, *args, **kwargs)
                    else:
                        result = method(*args, **kwargs)

                    if inspect.isawaitable(result):
                        fut = self._submit_to_loop(result)
                        fut.add_done_callback(
                            lambda f, en=event.name: logger.warning(f"Error in {en} callback: {f.exception()}")
                            if f.exception()
                            else None
                        )
            except Exception as e:
                logger.warning(f"Error in {event.name} callback: {e}")

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

    # Internal helper – used by LLMProgram.start_sync
    def _set_loop(self, loop: asyncio.AbstractEventLoop, thread: threading.Thread | None, own: bool = False) -> None:  # noqa: D401,E501
        """Assign an existing event loop & thread to this process."""
        self._loop = loop
        self._loop_thread = thread
        self._own_loop = own

    async def call_tool(self, tool_name: str, args_dict: dict) -> Any:
        """Call a tool by name with the given arguments dictionary.

        This method provides a unified interface for calling any registered tool,
        whether it's an MCP tool, a system tool, or a function-based tool.
        It delegates to the ToolManager which handles all tool calling details
        and performs file descriptor processing if needed.

        Args:
            tool_name: The name of the tool to call
            args_dict: Dictionary of arguments to pass to the tool

        Returns:
            The result of the tool execution or an error ToolResult
        """
        try:
            # Call the tool through tool manager
            result = await self.tool_manager.call_tool(tool_name, args_dict)

            # Log any errors that were returned
            if hasattr(result, "is_error") and result.is_error:
                logger.error(f"Tool '{tool_name}' returned error: {result.content}")
                return result

            # Process result for file descriptors if needed
            if self.file_descriptor_enabled and self.fd_manager and hasattr(result, "content"):
                processed_result, used_fd = self.fd_manager.create_fd_from_tool_result(result.content, tool_name)

                if used_fd:
                    logger.info(
                        f"Tool result from '{tool_name}' exceeds {self.fd_manager.max_direct_output_chars} chars, creating file descriptor"
                    )
                    result = processed_result
                    logger.debug(f"Created file descriptor for tool result from '{tool_name}'")

            return result
        except Exception as e:
            error_msg = f"Error calling tool '{tool_name}': {str(e)}"
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

    async def _fork_process(self, access_level: AccessLevel = AccessLevel.WRITE) -> "LLMProcess":
        """Create a deep copy of this process with preserved state.

        This implements the fork system call semantics where a copy of the
        process is created with the same state and configuration. The forked
        process is completely independent and can run separate tasks.

        This method follows the standard Unix pattern where:
        1. A new base process is created through the standard program_exec.create_process path
        2. Runtime state is copied from parent to child
        3. State-specific configurations are preserved
        4. Access level is set on the child process

        Args:
            access_level: Access level to set for the child process (defaults to WRITE)

        Returns:
            A new LLMProcess instance that is a deep copy of this one

        Raises:
            RuntimeError: If the current process doesn't have ADMIN access
        """
        # Check if the current process has permission to fork
        # Only processes with ADMIN access can create new child processes
        if not hasattr(self, "access_level") or self.access_level != AccessLevel.ADMIN:
            raise RuntimeError("Forking requires ADMIN access level and is not allowed for this process")

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

        # Copy runtime state using snapshot helper
        snapshot = ProcessSnapshot(
            state=copy.deepcopy(self.state),
            enriched_system_prompt=getattr(self, "enriched_system_prompt", None),
        )

        if hasattr(forked_process, "_apply_snapshot"):
            forked_process._apply_snapshot(snapshot)
        else:
            # degraded mode for heavily mocked objects in unit tests
            forked_process.state = snapshot.state
            forked_process.enriched_system_prompt = snapshot.enriched_system_prompt

        # Copy file descriptor manager state if FD is enabled
        # File descriptor system state needs special handling to ensure
        # complete independence between parent and child
        if getattr(self, "file_descriptor_enabled", False) and getattr(self, "fd_manager", None):
            forked_process.file_descriptor_enabled = True
            # Use subsystem‑provided clone helper
            try:
                forked_process.fd_manager = self.fd_manager.clone()
            except AttributeError:
                # Fallback to deepcopy if clone not yet implemented
                forked_process.fd_manager = copy.deepcopy(self.fd_manager)

            forked_process.references_enabled = getattr(self, "references_enabled", False)

        # Copy callbacks from parent to child
        if hasattr(self, "callbacks") and self.callbacks:
            forked_process.callbacks = self.callbacks.copy()

        # Set access level for the child process using process_access_level on tool_manager
        # This prevents the child from calling ADMIN tools like fork
        if hasattr(forked_process, "tool_manager"):
            forked_process.tool_manager.set_process_access_level(access_level)
            logger.debug(f"Set access level for forked process to {access_level.value}")

        # The child process will have its access level set according to the parameter,
        # which defaults to AccessLevel.WRITE, preventing it from calling fork again
        # (Only processes with ADMIN access level can use fork)

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

        logger.info(f"Fork successful. New process created for {forked_display} with {access_level.value} access")
        return forked_process

    async def fork_process(self, access_level: AccessLevel = AccessLevel.WRITE) -> "LLMProcess":
        """Create a deep copy of this process with preserved state.

        Deprecated: Use _fork_process instead. This method is kept for backward
        compatibility and will be removed in a future release.

        Args:
            access_level: Access level to set for the child process (defaults to WRITE)

        Returns:
            A new LLMProcess instance that is a deep copy of this one
        """
        warnings.warn(
            "fork_process is deprecated and will be removed in a future version. Use _fork_process instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self._fork_process(access_level)

    # ------------------------------------------------------------------
    # Snapshot helpers (internal)
    # ------------------------------------------------------------------

    def _apply_snapshot(self, snapshot: "ProcessSnapshot") -> None:
        """Replace this process's conversation state with *snapshot*.

        Intended for use by fork_process; not part of the public API.

        This is part of the fork tool implementation and helps maintain proper
        state isolation between parent and child processes by applying a
        snapshot of the parent's state to the child process.

        Args:
            snapshot: An immutable snapshot containing the conversation state to apply
        """
        # Shallow assignment is safe – snapshot already deep‑copied lists.
        self.state = snapshot.state
        if snapshot.enriched_system_prompt is not None:
            self.enriched_system_prompt = snapshot.enriched_system_prompt


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
