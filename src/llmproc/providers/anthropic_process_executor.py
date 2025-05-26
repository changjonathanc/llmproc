"""Anthropic provider tools implementation for LLMProc."""

import asyncio
import copy
import logging
import os
from typing import Any, Optional

# Import Anthropic clients (will be None if not installed)
try:
    from anthropic import AsyncAnthropic, AsyncAnthropicVertex
except ImportError:
    AsyncAnthropic = None
    AsyncAnthropicVertex = None

from llmproc.callbacks import CallbackEvent
from llmproc.common.results import RunResult, ToolResult
from llmproc.providers.anthropic_utils import (
    add_token_efficient_header_if_needed,
    get_context_window_size,
    prepare_api_request,
)
from llmproc.providers.constants import ANTHROPIC_PROVIDERS
from llmproc.providers.utils import safe_callback
from llmproc.utils.message_utils import append_message_with_id

# Set up logging
logger = logging.getLogger(__name__)

try:  # pragma: no cover - anthropic optional
    from anthropic import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        OverloadedError,
        RateLimitError,
    )
except Exception:  # pragma: no cover - anthropic optional

    class RateLimitError(Exception):
        pass

    class OverloadedError(Exception):
        pass

    class APIStatusError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass


async def _call_with_retry(client: Any, request: dict[str, Any]) -> Any:
    """Call ``client.messages.create`` with retries based on environment vars."""
    max_attempts = int(os.getenv("LLMPROC_RETRY_MAX_ATTEMPTS", "6"))
    initial_wait = int(os.getenv("LLMPROC_RETRY_INITIAL_WAIT", "1"))
    max_wait = int(os.getenv("LLMPROC_RETRY_MAX_WAIT", "90"))

    attempt = 0
    wait = initial_wait
    while True:
        try:
            return await client.messages.create(**request)
        except (
            RateLimitError,
            OverloadedError,
            APIStatusError,
            APIConnectionError,
            APITimeoutError,
        ) as e:
            attempt += 1
            if attempt >= max_attempts:
                logger.warning(
                    f"Max retry attempts ({max_attempts}) reached for Anthropic API call, giving up: {str(e)}"
                )
                raise
            logger.warning(f"Anthropic API error (attempt {attempt}/{max_attempts}), retrying in {wait}s: {str(e)}")
            await asyncio.sleep(min(wait, max_wait))
            wait = min(wait * 2, max_wait)


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
        process: "Process",  # noqa: F821
        user_prompt: str,
        max_iterations: int = 10,
        run_result: Optional["RunResult"] = None,
        is_tool_continuation: bool = False,
    ) -> "RunResult":
        """Execute a conversation with the Anthropic API.

        This method executes a conversation turn with proper tool handling, tracking metrics,
        and callback notifications. It can be used for both initial user messages and
        for continuing a conversation after tool execution.

        Args:
            process: The LLMProcess instance
            user_prompt: The user's input message
            max_iterations: Maximum number of API calls for tool usage
            run_result: Optional RunResult object to track execution metrics
            is_tool_continuation: Whether this is continuing a previous tool call

        Returns:
            RunResult object containing execution metrics and API call information
        """
        # Initialize run result if not provided
        if run_result is None:
            run_result = RunResult()

        if not is_tool_continuation:
            # Add user message with GOTO ID
            append_message_with_id(process, "user", user_prompt)

        process.run_stop_reason = None
        iterations = 0

        while iterations < max_iterations:
            # ── 1. reset per‑response buffers ───────────────────────────────────
            self.msg_prefix = []
            self.tool_results_prefix = []

            # Trigger TURN_START event
            process.trigger_event(CallbackEvent.TURN_START, process)

            # Set up runtime context with live references to our buffers
            ctx = process.tool_manager.runtime_context
            ctx["msg_prefix"] = self.msg_prefix
            ctx["tool_results_prefix"] = self.tool_results_prefix

            logger.debug(f"Making API call {iterations + 1}/{max_iterations}")

            # Prepare the API request with unified payload preparation
            use_caching = not getattr(process, "disable_automatic_caching", False)
            api_request = prepare_api_request(process, add_cache=use_caching)

            # Trigger API request event
            process.trigger_event(CallbackEvent.API_REQUEST, api_request)

            # Prepare and make API call with retry logic
            response = await _call_with_retry(process.client, api_request)

            # Trigger API response event
            process.trigger_event(CallbackEvent.API_RESPONSE, response)

            # Process API response

            # Track API call in the run result if available
            if run_result:
                api_info = {
                    "model": process.model_name,
                    "usage": getattr(response, "usage", {}),
                    "stop_reason": getattr(response, "stop_reason", None),
                    "id": getattr(response, "id", None),
                    "request": api_request,
                    "response": response,
                }
                run_result.add_api_call(api_info)

            stop_reason = response.stop_reason
            tool_invoked = False
            execution_aborted = False

            # ── 2. stream over content blocks ───────────────────────────────────
            for block in response.content:
                if block.type == "text":
                    # NOTE: sometimes model can decide to not respond with any text, for example, after using tools.
                    # appending the empty assistant message will cause the following API error in the next api call:
                    # ERROR: all messages must have non-empty content except for the optional final assistant message
                    if not hasattr(block, "text") or not block.text.strip():
                        continue  # Skip empty text blocks

                    # Trigger response event
                    process.trigger_event(CallbackEvent.RESPONSE, block.text)

                    # Store the original block for later assembly
                    self.msg_prefix.append(block)
                    continue

                if block.type != "tool_use":
                    continue  # Safety for future block types

                # Store the original block
                self.msg_prefix.append(block)

                # Extract tool details
                tool_name = block.name
                tool_args = block.input
                tool_id = block.id

                # Trigger tool_start event
                process.trigger_event(CallbackEvent.TOOL_START, tool_name, tool_args)

                # Track tool in run_result if available
                if run_result:
                    run_result.add_tool_call(tool_name, tool_args)

                # Execute tool ---------------------------------------------------
                # Set tool_id for this specific tool call
                ctx["tool_id"] = tool_id

                # Call the tool
                logger.debug(f"Calling tool '{tool_name}' with parameters: {tool_args}")
                result = await process.call_tool(tool_name, tool_args)

                # Remove tool_id from context now that the call is complete
                ctx.pop("tool_id", None)

                # Trigger tool_end event
                process.trigger_event(CallbackEvent.TOOL_END, tool_name, result)

                # Check if tool execution should abort further processing
                if hasattr(result, "abort_execution") and result.abort_execution:
                    logger.info(
                        f"Tool '{tool_name}' requested execution abort. Stopping tool processing for this response."
                    )
                    execution_aborted = True
                    break  # Exit the loop processing tools for this API response

                # Process result for file descriptors if needed
                if not isinstance(result, ToolResult):
                    # This is a programming error - tools must return ToolResult
                    error_msg = (
                        f"Tool '{tool_name}' did not return a ToolResult instance. Got {type(result).__name__} instead."
                    )
                    logger.error(error_msg)
                    tool_result = ToolResult.from_error(error_msg)
                else:
                    tool_result = result

                # Create tool result content for state
                tool_result_dict = tool_result.to_dict()
                tool_result_content = {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    **tool_result_dict,
                }

                # Add to tool_results_prefix for causal history tracking
                self.tool_results_prefix.append(tool_result_content)
                tool_invoked = True

            # ── 3. commit this provider response to conversation state ─────────
            # Only update state if execution was not aborted by a tool
            if not execution_aborted:
                # Add assistant message with all content blocks
                if self.msg_prefix:
                    append_message_with_id(process, "assistant", self.msg_prefix)

                # Add tool results as user messages
                if self.tool_results_prefix:
                    for tool_result in self.tool_results_prefix:
                        append_message_with_id(process, "user", tool_result)

            # Trigger TURN_END event
            process.trigger_event(
                CallbackEvent.TURN_END,
                process,
                response,
                self.tool_results_prefix,
            )

            # If no response or no tools were invoked, we're done with this iteration
            if not response.content or not tool_invoked:
                # Get out of the tool loop as there are no more tools to execute
                # Note: response.content could be empty in rare cases, which we handle gracefully
                process.run_stop_reason = stop_reason
                break

            iterations += 1

        if iterations >= max_iterations:
            process.run_stop_reason = "max_iterations"

        # Create a new RunResult if one wasn't provided
        if run_result is None:
            run_result = RunResult()

        # Set the last_message in the RunResult to ensure it's available
        # This is critical for the sync interface tests
        last_message = process.get_last_message()
        run_result.set_last_message(last_message)

        # Complete the RunResult and return it
        return run_result.complete()

    async def count_tokens(self, process: "Process") -> dict:
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
