"""LLMProcess class for executing LLM programs and handling interactions."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from llmproc.callbacks import CallbackEvent
from llmproc.common.results import RunResult, ToolResult
from llmproc.config.process_config import ProcessConfig
from llmproc.event_loop_mixin import EventLoopMixin
from llmproc.plugin.plugin_event_runner import PluginEventRunner
from llmproc.plugin.protocol import PluginProtocol
from llmproc.plugins.stderr import StderrPlugin
from llmproc.process_forking import ProcessForkingMixin
from llmproc.providers.utils import choose_provider_executor

# Set up logger
logger = logging.getLogger(__name__)


_T = TypeVar("_T")


class LLMProcess(EventLoopMixin, ProcessForkingMixin):
    """Process for interacting with LLMs using standardized program definitions."""

    def __init__(self, cfg: ProcessConfig) -> None:
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
            cfg: Configuration object with all initialization parameters

        Raises:
            ValueError: If required parameters are missing
        """
        # Basic validation for required parameters
        if not cfg.model_name or not cfg.provider:
            raise ValueError("model_name and provider are required for LLMProcess initialization")

        # Initialize event loop handling
        EventLoopMixin.__init__(self, cfg.loop)

        # Store all provided state attributes
        self.program = cfg.program
        self.model_name = cfg.model_name
        self.provider = cfg.provider
        self.base_system_prompt = cfg.base_system_prompt
        self.base_dir = cfg.base_dir
        self.api_params = cfg.api_params or {}
        self.parameters = {}  # Parameters are already processed in program

        # Runtime state
        self.state = cfg.state or []
        self.enriched_system_prompt = cfg.enriched_system_prompt

        # Per-iteration buffers managed by executors
        self.iteration_state = None

        # Client
        self.client = cfg.client

        # Initialize provider-specific executor
        self.executor = choose_provider_executor(cfg.provider, cfg.model_name)

        # Tool management and access control
        self.tool_manager = cfg.tool_manager
        self.access_level = cfg.access_level
        if cfg.tool_manager:
            self.tool_manager.set_process_access_level(cfg.access_level)

        # MCP configuration
        self.mcp_config_path = cfg.mcp_config_path
        self.mcp_servers = cfg.mcp_servers
        self.mcp_tools = cfg.mcp_tools or {}
        self.mcp_enabled = (
            cfg.mcp_enabled
            if cfg.mcp_enabled is not None
            else (cfg.mcp_config_path is not None or cfg.mcp_servers is not None)
        )

        # User prompt configuration
        self.user_prompt = cfg.user_prompt
        self.max_iterations = cfg.max_iterations

        # Unified plugin runner
        self.plugins = PluginEventRunner(self._submit_to_loop, cfg.plugins or [])

    def add_plugins(self, *plugins: Callable) -> "LLMProcess":
        """Register plugins with the unified plugin runner."""
        for plugin in plugins:
            self.plugins.add(plugin)
        return self

    def get_plugin(self, plugin_type: type) -> Any | None:
        """Return the first plugin instance matching ``plugin_type``.

        This is an internal helper for tests and plugins and is not part of the
        public API.
        """
        for plugin in self.plugins:
            if isinstance(plugin, plugin_type):
                return plugin
        return None

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - defensive
        """Fallback attribute handler."""
        raise AttributeError(name)

    # ------------------------------------------------------------------
    # Private event-loop helpers
    # ------------------------------------------------------------------

    async def trigger_event(self, event: CallbackEvent, **kwargs) -> None:
        """Trigger an event to all registered plugins."""
        await self.plugins.run_event(event.value, self, **kwargs)

    def _process_user_input(self, user_input: str) -> str:
        """Validate user input."""
        if not user_input or user_input.strip() == "":
            raise ValueError("User input cannot be empty")
        return user_input

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

        # Apply user input hooks
        hooked_user_input = await self.plugins.user_input(user_input, self)

        processed_user_input = self._process_user_input(hooked_user_input)
        run_result = await self.executor.run(self, processed_user_input, max_iterations)
        await self.trigger_event(CallbackEvent.RUN_END, run_result=run_result)
        return run_result

    def get_state(self) -> list[dict[str, str]]:
        """Return the current conversation state.

        Returns:
            A copy of the current conversation state
        """
        return self.state.copy()

    def reset_state(self) -> None:
        """Reset the conversation state.

        Note:
            State only contains user and assistant messages. The system prompt
            and preloaded content are immutable and always preserved. Plugins
            are responsible for managing their own state during reset.
        """
        # Clear the conversation state (user/assistant messages)
        self.state = []

    @property
    def tools(self) -> list:
        """Property to access tool definitions for the LLM API.

        This delegates to the ToolManager which provides a consistent interface
        for getting tool schemas across all tool types.

        The ToolManager handles filtering and validation.

        Returns:
            List of tool schemas formatted for the LLM provider's API.
        """
        # Get schemas from the tool manager
        # This includes filtering for enabled tools
        return self.tool_manager.get_tool_schemas()

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
            aggregator = getattr(self.tool_manager, "mcp_aggregator", None)
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

    def __init__(self, cfg: ProcessConfig, _loop: Optional[asyncio.AbstractEventLoop] = None):
        """Initialize with parameters from program_exec.create_sync_process.

        ⚠️ WARNING: DO NOT USE THIS CONSTRUCTOR DIRECTLY! ⚠️

        ALWAYS use the synchronous factory method `program.start_sync()` instead,
        which properly handles initialization following the Unix-inspired pattern:

        ```python
        program = LLMProgram.from_toml("config.toml")
        process = program.start_sync()  # CORRECT WAY TO CREATE SYNCLLMPROCESS
        ```

        Args:
            cfg: Configuration object with initialization parameters
            _loop: Optional event loop to use (creates a new one if not provided)
        """
        super().__init__(cfg)

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
