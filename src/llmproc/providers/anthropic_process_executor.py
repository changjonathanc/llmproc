"""Anthropic provider tools implementation for LLMProc."""

import copy
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

# Import Anthropic clients (will be None if not installed)
try:
    from anthropic import AsyncAnthropic, AsyncAnthropicVertex
except ImportError:
    AsyncAnthropic = None
    AsyncAnthropicVertex = None

from llmproc.callbacks import CallbackEvent
from llmproc.common.results import RunResult
from llmproc.providers.anthropic_utils import (
    caching_disabled,
    prepare_api_request,
    stream_call_with_retry,
)
from llmproc.providers.utils import get_context_window_size
from llmproc.utils.background import AsyncBackgroundIterator
from llmproc.utils.message_utils import append_message

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from llmproc.llm_process import LLMProcess

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class IterationState:
    """Mutable values for a single API iteration."""

    msg_prefix: list = field(default_factory=list)
    tool_results_prefix: list = field(default_factory=list)
    current_tool: Any | None = None
    execution_aborted: bool = False
    commit_partial: bool = True


class AnthropicProcessExecutor:
    """Process executor for Anthropic models.

    This class manages interactions with the Anthropic API, including
    handling conversation flow, tool calls, and response processing.
    """

    # Map of model names to context window sizes
    CONTEXT_WINDOW_SIZES = {
        "claude-3-5-sonnet": 200000,
        "claude-3-5-haiku": 200000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-3-7-sonnet": 200000,
    }

    #
    # Primary methods
    #

    async def run(
        self,
        process: "LLMProcess",
        user_prompt: str | None = None,
        max_iterations: int = 10,
        is_tool_continuation: bool = False,
    ) -> "RunResult":
        """Execute a conversation with the Anthropic API.

        This method executes a conversation turn with proper tool handling, tracking metrics,
        and callback notifications. It can be used for both initial user messages and
        for continuing a conversation after tool execution.

        Args:
            process: The LLMProcess instance
            user_prompt: The user's input message (required unless continuing a tool call)
            max_iterations: Maximum number of API calls for tool usage
            is_tool_continuation: Whether this is continuing a previous tool call

        Returns:
            RunResult object containing execution metrics and API call information
        """
        run_result = RunResult()

        if not is_tool_continuation:
            if user_prompt is None:
                raise ValueError("user_prompt is required when not continuing a tool call")
            append_message(process, "user", user_prompt)

        run_result.set_stop_reason(None)
        iterations = 0

        while iterations < max_iterations:
            state = IterationState()
            process.iteration_state = state

            await process.trigger_event(CallbackEvent.TURN_START, run_result=run_result)

            logger.debug(f"Making API call {iterations + 1}/{max_iterations}")

            api_request = await self._prepare_request(process)
            block_gen = await self._send_request(process, api_request)

            tool_invoked, response = await self._stream_blocks(process, block_gen, run_result, state)

            await process.trigger_event(CallbackEvent.API_RESPONSE, response=response)

            api_info = {
                "model": process.model_name,
                "usage": getattr(response, "usage", {}),
                "stop_reason": getattr(response, "stop_reason", None),
                "id": getattr(response, "id", None),
                "request": api_request,
                "response": response,
            }
            run_result.add_api_call(api_info)

            stop_reason = getattr(response, "stop_reason", None)

            await self._commit_state(process, response, state)

            if state.execution_aborted:
                run_result.set_stop_reason("hook_stop")
                break

            if not getattr(response, "content", None) or not tool_invoked:
                run_result.set_stop_reason(stop_reason)
                break

            iterations += 1

        if iterations >= max_iterations:
            run_result.set_stop_reason("max_iterations")

        # Set the last_message in the RunResult to ensure it's available
        # This is critical for the sync interface tests
        last_message = process.get_last_message()
        run_result.set_last_message(last_message)

        # Complete the RunResult and return it
        return run_result.complete()

    async def _prepare_request(self, process: "LLMProcess") -> dict[str, Any]:
        """Prepare Anthropic API request and trigger event."""
        use_caching = not caching_disabled()
        api_request = prepare_api_request(process, add_cache=use_caching)
        await process.trigger_event(CallbackEvent.API_REQUEST, api_request=api_request)
        return api_request

    async def _send_request(self, process: "LLMProcess", api_request: dict[str, Any]):
        """Send request to Anthropic and yield streaming blocks."""
        return stream_call_with_retry(process.client, api_request)

    async def _stream_blocks(
        self,
        process: "LLMProcess",
        block_generator,
        run_result: RunResult,
        state: IterationState,
    ) -> tuple[bool, Any]:
        """Stream content blocks and execute tools."""
        tool_invoked = False
        response_obj = None

        async def _immediate_streaming_callback(block):
            """Execute streaming callbacks immediately when blocks arrive."""
            await process.trigger_event(CallbackEvent.API_STREAM_BLOCK, block=block)

        async with AsyncBackgroundIterator(
            block_generator,
            on_item=_immediate_streaming_callback,
        ) as blocks:
            async for block in blocks:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    if not hasattr(block, "text") or not block.text.strip():
                        continue
                    hook_res = await process.plugins.response(process, block.text)
                    if hook_res is not None and getattr(hook_res, "stop", False):
                        state.commit_partial = getattr(hook_res, "commit_current", True)
                        if state.commit_partial:
                            state.msg_prefix.append(block)
                        else:
                            state.msg_prefix.clear()
                        state.execution_aborted = True
                        break
                    state.msg_prefix.append(block)
                    continue

                if block_type == "thinking":
                    # Handle thinking blocks - add to message prefix but don't treat as final response
                    state.msg_prefix.append(block)
                    continue

                if block_type != "tool_use":
                    response_obj = block
                    break
                state.msg_prefix.append(block)
                invoked, aborted = await self._execute_tool(process, block, run_result, state)
                tool_invoked = tool_invoked or invoked
                if aborted:
                    state.execution_aborted = True
                    continue

        if response_obj is None:
            response_obj = SimpleNamespace(content=[], stop_reason=None, id=None, usage=SimpleNamespace())
        return tool_invoked, response_obj

    async def _execute_tool(
        self,
        process: "LLMProcess",
        block: Any,
        run_result: RunResult,
        state: IterationState,
    ) -> tuple[bool, bool]:
        """Execute a single tool_use block."""
        tool_name = block.name
        tool_args = block.input
        tool_id = block.id

        await process.trigger_event(CallbackEvent.TOOL_START, tool_name=tool_name, tool_args=tool_args)
        run_result.add_tool_call(tool_name=tool_name, tool_args=tool_args)

        state.current_tool = block
        logger.debug(f"Calling tool '{tool_name}' with parameters: {tool_args}")
        result = await process.call_tool(tool_name, tool_args)
        state.current_tool = None

        await process.trigger_event(CallbackEvent.TOOL_END, tool_name=tool_name, result=result)

        # Always create tool_result to maintain API protocol compliance
        tool_result_dict = result.to_dict()
        tool_result_content = {
            "type": "tool_result",
            "tool_use_id": tool_id,
            **tool_result_dict,
        }
        state.tool_results_prefix.append(tool_result_content)

        if hasattr(result, "abort_execution") and result.abort_execution:
            logger.info(f"Tool '{tool_name}' requested execution abort. Stopping tool processing for this response.")
            return True, True

        return True, False

    async def _commit_state(self, process: "LLMProcess", response: Any, state: IterationState) -> None:
        """Commit streamed content and tool results to process state."""
        if not state.execution_aborted or state.commit_partial:
            if state.msg_prefix:
                append_message(process, "assistant", state.msg_prefix)

        # Always commit tool results to maintain API protocol compliance
        # Even when execution is aborted, tool_result blocks must be present
        if state.tool_results_prefix:
            for tool_result in state.tool_results_prefix:
                append_message(process, "user", tool_result)

        await process.trigger_event(
            CallbackEvent.TURN_END,
            response=response,
            tool_results=state.tool_results_prefix,
        )

        process.iteration_state = None

    async def count_tokens(self, process: "LLMProcess") -> dict:
        """Count tokens in the current conversation context using Anthropic's API."""
        try:
            # Create state copy with dummy message and prepare API request
            process_copy = copy.copy(process)
            process_copy.state = copy.deepcopy(process.state or []) + [{"role": "user", "content": "Hi"}]
            api_request = prepare_api_request(process_copy, add_cache=False)

            # Get token count with inline parameter validation
            system = api_request.get("system")
            tools = api_request.get("tools")
            response = await process.client.messages.count_tokens(
                model=process_copy.model_name,
                messages=api_request["messages"],
                **({"system": system} if isinstance(system, list) and system else {}),
                **({"tools": tools} if isinstance(tools, list) and tools else {}),
            )

            # Calculate window metrics
            tokens = getattr(response, "input_tokens", 0)
            window_size = get_context_window_size(process.model_name, self.CONTEXT_WINDOW_SIZES)

            return {
                "input_tokens": tokens,
                "context_window": window_size,
                "percentage": (tokens / window_size * 100) if window_size > 0 else 0,
                "remaining_tokens": max(0, window_size - tokens),
            }
        except Exception as e:
            logger.warning(f"Token counting failed: {str(e)}")
            return {"error": str(e)}
