"""OpenAI Responses API executor for LLMProc.

This executor handles OpenAI reasoning models using the Responses API, including:
- o1, o3, o4 series models
- Models that benefit from preserved reasoning context across tool calls

For non-reasoning models (GPT-4, GPT-4o, GPT-3.5), use the openai_chat provider
which implements the Chat Completions API.

The Responses API manages conversation state server-side via response_id references,
allowing for more efficient reasoning model interactions while maintaining
compatibility with LLMProc's client-side state management.
"""

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from llmproc.callbacks import CallbackEvent
from llmproc.common.results import RunResult
from llmproc.providers.openai_utils import (
    call_with_retry,
    convert_tools_to_openai_format,
    format_tool_result_for_openai,
)
from llmproc.utils.message_utils import append_message

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from llmproc.llm_process import LLMProcess

logger = logging.getLogger(__name__)


def _normalize_responses_params(api_params: dict[str, Any]) -> dict[str, Any]:
    """Normalize parameters for Responses API."""
    params = api_params.copy()

    # Remove parameters that are not supported by Responses API
    # Based on OpenAI Responses API documentation, some parameters like max_tokens may not be supported
    unsupported_params = {"max_tokens", "max_completion_tokens", "temperature"}
    for param in unsupported_params:
        if param in params:
            params.pop(param)

    # Handle reasoning parameters
    reasoning_config = {}
    if "reasoning_effort" in params:
        reasoning_config["effort"] = params.pop("reasoning_effort")
    if "reasoning_summary" in params:
        reasoning_config["summary"] = params.pop("reasoning_summary")
    else:
        # Only set default if we have reasoning effort
        if "effort" in reasoning_config:
            reasoning_config["summary"] = "auto"  # Default

    if reasoning_config:
        params["reasoning"] = reasoning_config

    return params


class OpenAIResponseProcessExecutor:
    """Process executor for OpenAI Responses API.

    This executor implements conversation flow and tool usage for the
    OpenAI Responses API, designed for reasoning models (o1, o3, o4, etc.).

    Key differences from Chat Completions API:
    - Uses input/previous_response_id instead of messages array
    - Stores response_id metadata in conversation state
    - Optimized for reasoning model context preservation

    For non-reasoning models, use OpenAIProcessExecutor instead.
    """

    async def run(
        self,
        process: "LLMProcess",
        user_prompt: str | None = None,
        max_iterations: int = 10,
        is_tool_continuation: bool = False,
    ) -> RunResult:
        """Execute a conversation with the OpenAI Responses API.

        Args:
            process: The LLMProcess instance
            user_prompt: The user's input message (required unless continuing a tool call)
            max_iterations: Maximum number of tool iterations
            is_tool_continuation: Whether this call continues after a tool result

        Returns:
            RunResult object containing execution metrics and API call information

        Raises:
            ValueError: If the OpenAI Responses API call fails
        """
        run_result = RunResult()

        # Always add user message to state first (unless continuing)
        if not is_tool_continuation:
            if user_prompt is None:
                raise ValueError("user_prompt is required when not continuing a tool call")
            append_message(process, "user", user_prompt)

        iterations = 0
        while iterations < max_iterations:
            logger.debug(f"Making OpenAI Responses API call {iterations + 1}/{max_iterations}")

            # Trigger TURN_START event
            await process.trigger_event(CallbackEvent.TURN_START, run_result=run_result)

            try:
                # ── 1. Get conversation state for API call ──────────────────────────
                last_response_id, messages_since_response = self._get_conversation_payload(process)

                # ── 2. Build API call parameters ────────────────────────────────────
                api_params = _normalize_responses_params(process.api_params)
                responses_tools = convert_tools_to_openai_format(process.tools, api_type="responses")

                call_params = {
                    "model": process.model_name,
                    **api_params,
                }

                if responses_tools:
                    call_params["tools"] = responses_tools

                if last_response_id:
                    # Continuation: send previous response_id + all messages since then
                    call_params["previous_response_id"] = last_response_id
                    call_params["input"] = messages_since_response
                else:
                    # New conversation: send just the current user message as properly formatted input
                    call_params["input"] = [{"type": "message", "role": "user", "content": user_prompt}]

                # Build API request payload for logging
                api_request = {
                    "model": process.model_name,
                    "params": call_params,
                }

                # Trigger API request event
                await process.trigger_event(CallbackEvent.API_REQUEST, api_request=api_request)

                # ── 3. Make API call ─────────────────────────────────────────────────
                response = await call_with_retry(process.client, "responses", call_params)

                # Trigger API response event
                await process.trigger_event(CallbackEvent.API_RESPONSE, response=response)

                # Track API call in the run result
                api_info = {
                    "model": process.model_name,
                    "usage": getattr(response, "usage", {}),
                    "id": getattr(response, "id", None),
                    "request": api_request,
                    "response": response,
                }
                run_result.add_api_call(api_info)

                # ── 4. Process response and commit to state ──────────────────────────
                # Store complete response object in conversation state
                self._add_response_to_state(process, response)

                # Process response content and add assistant messages, tool results
                tool_calls_made, stopped = await self._process_response_outputs(process, response, run_result)

                # Trigger TURN_END event
                await process.trigger_event(CallbackEvent.TURN_END, response=response, tool_results=[])

                if stopped:
                    run_result.set_stop_reason("hook_stop")
                    break

                if not tool_calls_made:
                    # No tool calls, conversation is complete
                    break

                iterations += 1

            except Exception as e:
                logger.error(f"Error in OpenAI Responses API call: {str(e)}")
                # Add error to run result
                run_result.add_api_call({"type": "error", "error": str(e)})
                run_result.set_stop_reason("error")
                raise

        # Set the last_message in the RunResult
        run_result.last_message = process.get_last_message()

        # Complete the RunResult and return it
        return run_result.complete()

    async def _process_response_outputs(
        self,
        process: "LLMProcess",
        response: Any,
        run_result: RunResult,  # noqa: F821
    ) -> tuple[bool, bool]:
        """Process response outputs.

        Args:
            process: The LLMProcess instance
            response: The OpenAI Responses API response
            run_result: The RunResult object to track tool calls

        Returns:
            tuple: (tool_calls_made, stopped)
        """
        tool_calls_made = False
        tool_results = []

        # Process each output item from the response
        for item in response.output:
            logger.debug(f"Processing response item: type={item.type}")

            if item.type == "reasoning":
                # Skip reasoning items for now - they contain the model's internal reasoning
                continue

            elif item.type == "message":
                # Assistant message with content array
                for content_item in item.content:
                    if content_item.type == "output_text":
                        logger.debug(f"Adding assistant message: {content_item.text}")
                        append_message(process, "assistant", content_item.text)
                        hook_res = await process.plugins.response(process, content_item.text)
                        if hook_res is not None and getattr(hook_res, "stop", False):
                            if not getattr(hook_res, "commit_current", True):
                                process.state.pop()
                            run_result.set_stop_reason("hook_stop")
                            return tool_calls_made, True

            elif item.type == "function_call":
                # Tool call to execute
                tool_calls_made = True
                result = await self._execute_tool_call(process, item, run_result)
                tool_results.append(result)

        # Note: We don't create tool continuation here - that's handled by the main loop
        # The tool results will be sent in the next iteration via previous_response_id

        return tool_calls_made, False

    def _get_conversation_payload(self, process: "LLMProcess") -> tuple[str | None, list]:
        """Get the last response_id and all messages since that response for API payload.

        Args:
            process: The LLMProcess instance

        Returns:
            tuple: (last_response_id, messages_since_response)
                   - last_response_id: str or None
                   - messages_since_response: list of formatted messages for API
        """
        last_response_id = None
        last_response_idx = -1

        # Find the most recent response object
        for i, msg in enumerate(process.state):
            if msg.get("role") == "openai_response":
                response_obj = msg.get("response")
                if response_obj and hasattr(response_obj, "id"):
                    last_response_id = response_obj.id
                    last_response_idx = i

        if last_response_id is None:
            # No previous responses - new conversation
            return None, []

        # Collect all messages after the last response
        messages_since_response = []
        for i in range(last_response_idx + 1, len(process.state)):
            msg = process.state[i]

            if msg.get("role") == "user":
                # User message -> send as message input with proper formatting
                messages_since_response.append({"type": "message", "role": "user", "content": msg.get("content", "")})

            elif msg.get("role") == "tool":
                # Tool result -> format for function_call_output
                messages_since_response.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", "unknown"),
                        "output": msg.get("content", ""),
                    }
                )

        return last_response_id, messages_since_response

    def _add_response_to_state(self, process: "LLMProcess", response: Any) -> None:
        """Add complete response object to conversation state for audit trail.

        Args:
            process: The LLMProcess instance
            response: The OpenAI Responses API response object
        """
        response_entry = {
            "role": "openai_response",
            "response": response,
            "response_id": response.id,
            "timestamp": time.time(),
            "api_type": "responses",
        }
        process.state.append(response_entry)

    async def _execute_tool_call(
        self,
        process: "LLMProcess",
        call_item: Any,
        run_result: RunResult,  # noqa: F821
    ) -> dict[str, Any]:
        """Execute a single tool call from Responses API output.

        Args:
            process: The LLMProcess instance
            call_item: The function_call item from response.output
            run_result: The RunResult object to track the call

        Returns:
            Dictionary containing the formatted tool result for Responses API
        """
        name = call_item.name
        call_id = call_item.call_id

        try:
            args_dict = json.loads(call_item.arguments)
        except Exception:
            args_dict = {}

        await process.trigger_event(CallbackEvent.TOOL_START, tool_name=name, tool_args=args_dict)
        run_result.add_tool_call(tool_name=name, tool_args=args_dict)

        result = await process.call_tool(name, args_dict)
        await process.trigger_event(CallbackEvent.TOOL_END, tool_name=name, result=result)

        # Add to conversation state (for local tracking)
        formatted_content = (
            f"ERROR: {result.to_dict().get('content', '')}" if result.is_error else result.to_dict().get("content", "")
        )
        append_message(process, "tool", formatted_content)
        process.state[-1]["tool_call_id"] = call_id

        # Return formatted for Responses API
        return format_tool_result_for_openai(result, call_id=call_id, api_type="responses")

    async def count_tokens(self, process: "LLMProcess") -> dict:
        """Count tokens in the current conversation.

        Args:
            process: The LLMProcess instance

        Returns:
            Dictionary containing token usage information
        """
        try:
            # Import OpenAI utilities for token counting
            from llmproc.providers.openai_utils import CONTEXT_WINDOW_SIZES, num_tokens_from_messages
            from llmproc.providers.utils import get_context_window_size

            # Filter out response_metadata entries for token counting
            messages: list[dict[str, Any]] = []
            if process.enriched_system_prompt:
                messages.append({"role": "system", "content": process.enriched_system_prompt})

            # Filter state to exclude metadata and response objects
            filtered_state = [
                msg for msg in process.state if msg.get("role") not in ("response_metadata", "openai_response")
            ]
            messages.extend(filtered_state)

            tokens = num_tokens_from_messages(messages, model=process.model_name)
            window_size = get_context_window_size(process.model_name, CONTEXT_WINDOW_SIZES)

            return {
                "input_tokens": tokens,
                "context_window": window_size,
                "percentage": (tokens / window_size * 100) if window_size > 0 else 0,
                "remaining_tokens": max(0, window_size - tokens),
                "note": "Token counting for Responses API filters out metadata and response object entries",
            }
        except Exception as exc:
            logger.warning(f"Token counting failed: {exc}")
            return {"error": str(exc)}
