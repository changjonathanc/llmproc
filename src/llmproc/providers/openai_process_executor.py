"""OpenAI provider implementation for LLMProc.

This executor handles OpenAI models using the Chat Completions API, including:
- GPT-4, GPT-4o, GPT-3.5 models
- Non-reasoning models that use traditional chat format

This executor is used for:
- provider = "openai" with non-reasoning models (auto-selected)
- provider = "openai_chat" (explicit Chat Completions API)

For reasoning models (o1, o3, o4 series), the openai_response provider
uses the OpenAI Responses API (when implemented).

NOTE: This implementation is minimally maintained as we plan to integrate with LiteLLM
in a future release for more comprehensive provider support once Anthropic and core
functionality are mature enough.

Caching of the system prompt and recent messages is handled automatically by the
OpenAI API, so there's no need for ephemeral cache control like we do in the
Anthropic implementation.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

from llmproc.callbacks import CallbackEvent
from llmproc.common.results import RunResult
from llmproc.providers.openai_utils import (
    CONTEXT_WINDOW_SIZES,
    call_with_retry,
    convert_tools_to_openai_format,
    format_tool_result_for_openai,
    num_tokens_from_messages,
)
from llmproc.providers.utils import get_context_window_size
from llmproc.utils.message_utils import append_message

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from llmproc.llm_process import LLMProcess

logger = logging.getLogger(__name__)


def _format_state_messages(process: Any) -> list[dict[str, Any]]:
    """Convert process state to OpenAI chat message format."""
    messages: list[dict[str, Any]] = []

    if process.enriched_system_prompt:
        messages.append({"role": "system", "content": process.enriched_system_prompt})

    for message in process.state:
        role = message.get("role")
        if role == "assistant":
            msg = {"role": "assistant", "content": message.get("content")}
            if "tool_calls" in message:
                msg["tool_calls"] = message["tool_calls"]
            messages.append(msg)
        elif role == "user":
            messages.append({"role": "user", "content": message.get("content")})
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id"),
                    "content": message.get("content"),
                }
            )

    return messages


def _normalize_api_params(model_name: str, api_params: dict[str, Any]) -> dict[str, Any]:
    """Normalize API parameters based on model capabilities."""
    params = api_params.copy()

    is_reasoning_model = model_name.startswith(("o1", "o3"))
    if is_reasoning_model:
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
    else:
        params.pop("reasoning_effort", None)

    return params


class OpenAIProcessExecutor:
    """Process executor for OpenAI models.

    This executor implements basic conversation flow and tool usage for the
    OpenAI chat API. It is minimally maintained as the project plans to migrate
    to LiteLLM for unified provider support in the future.
    """

    async def run(
        self,
        process: "LLMProcess",
        user_prompt: str | None = None,
        max_iterations: int = 10,
        is_tool_continuation: bool = False,
    ) -> RunResult:
        """Execute a conversation with the OpenAI API.

        Args:
            process: The LLMProcess instance
            user_prompt: The user's input message (required unless continuing a tool call)
            max_iterations: Maximum number of tool iterations
            is_tool_continuation: Whether this call continues after a tool result

        Returns:
            RunResult object containing execution metrics and API call information

        Raises:
            ValueError: If the OpenAI API call fails
        """
        run_result = RunResult()

        if not is_tool_continuation:
            if user_prompt is None:
                raise ValueError("user_prompt is required when not continuing a tool call")
            append_message(process, "user", user_prompt)

        iterations = 0

        while iterations < max_iterations:
            logger.debug(f"Making OpenAI API call {iterations + 1}/{max_iterations}")

            # Trigger TURN_START event
            await process.trigger_event(CallbackEvent.TURN_START, run_result=run_result)

            formatted_messages = _format_state_messages(process)

            logger.debug(f"Making OpenAI API call with {len(formatted_messages)} messages")

            try:
                api_params = _normalize_api_params(process.model_name, process.api_params)

                openai_tools = convert_tools_to_openai_format(process.tools)

                # Build API request payload
                api_request = {
                    "model": process.model_name,
                    "messages": formatted_messages,
                    "params": api_params,
                }
                if openai_tools:
                    api_request["tools"] = openai_tools

                # Trigger API request event
                await process.trigger_event(CallbackEvent.API_REQUEST, api_request=api_request)

                # Make API call
                call_params = {
                    "model": process.model_name,
                    "messages": formatted_messages,
                    **api_params,
                }
                if openai_tools:
                    call_params["tools"] = openai_tools

                response = await call_with_retry(process.client, "chat", call_params)

                # Trigger API response event
                await process.trigger_event(CallbackEvent.API_RESPONSE, response=response)

                # Process API response

                # Track API call in the run result
                api_info = {
                    "model": process.model_name,
                    "usage": getattr(response, "usage", {}),
                    "id": getattr(response, "id", None),
                    "request": api_request,
                    "response": response,
                }
                run_result.add_api_call(api_info)

                # Extract the response message and any tool calls
                choice = response.choices[0]
                message = choice.message
                message_content = getattr(message, "content", "")
                finish_reason = choice.finish_reason

                # Set stop reason
                run_result.set_stop_reason(finish_reason)

                tool_calls = getattr(message, "tool_calls", None)
                if not isinstance(tool_calls, list):
                    tool_calls = []

                assistant_entry = {"role": "assistant", "content": message_content}
                if tool_calls:
                    serialized_calls = []
                    for call in tool_calls:
                        serialized_calls.append(
                            {
                                "id": getattr(call, "id", None),
                                "type": getattr(call, "type", "function"),
                                "function": {
                                    "name": getattr(call.function, "name", ""),
                                    "arguments": getattr(call.function, "arguments", "{}"),
                                },
                            }
                        )
                    assistant_entry["tool_calls"] = serialized_calls

                append_message(process, "assistant", message_content)
                if tool_calls:
                    process.state[-1]["tool_calls"] = serialized_calls

                # Trigger response event and hooks
                if message_content:
                    hook_res = await process.plugins.response(process, message_content)
                    if hook_res is not None and getattr(hook_res, "stop", False):
                        if not getattr(hook_res, "commit_current", True):
                            process.state.pop()
                        run_result.set_stop_reason("hook_stop")
                        break

                tool_results = []
                for call in tool_calls:
                    name = getattr(call.function, "name", "")
                    args_str = getattr(call.function, "arguments", "{}")
                    try:
                        args_dict = json.loads(args_str)
                    except Exception:  # noqa: BLE001 - fallback on parse errors
                        args_dict = {}

                    await process.trigger_event(CallbackEvent.TOOL_START, tool_name=name, tool_args=args_dict)
                    run_result.add_tool_call(tool_name=name, tool_args=args_dict)
                    result = await process.call_tool(name, args_dict)
                    await process.trigger_event(CallbackEvent.TOOL_END, tool_name=name, result=result)

                    tool_results.append(result.to_dict())
                    # OpenAI doesn't support the is_error field like Anthropic,
                    # so we format errors with "ERROR:" prefix for clear indication
                    formatted_content = format_tool_result_for_openai(result)
                    append_message(process, "tool", formatted_content)
                    process.state[-1]["tool_call_id"] = getattr(call, "id", None)

                # Trigger TURN_END event
                await process.trigger_event(CallbackEvent.TURN_END, response=response, tool_results=tool_results)

                # If no tool calls, we're done
                if not tool_calls:
                    break

                # Increment iteration counter for tool calls
                iterations += 1

            except Exception as e:
                logger.error(f"Error in OpenAI API call: {str(e)}")
                # Add error to run result
                run_result.add_api_call({"type": "error", "error": str(e)})
                run_result.set_stop_reason("error")
                raise

        # Set the last_message in the RunResult to ensure it's available
        # This is critical for the sync interface tests
        last_message = process.get_last_message()
        run_result.set_last_message(last_message)

        # Complete the RunResult and return it
        return run_result.complete()

    async def count_tokens(self, process: "LLMProcess") -> dict:
        """Count tokens in the current conversation using ``tiktoken``.

        Args:
            process: The ``LLMProcess`` instance.

        Returns:
            Dictionary containing token usage information.
        """
        try:
            messages: list[dict[str, Any]] = []
            if process.enriched_system_prompt:
                messages.append({"role": "system", "content": process.enriched_system_prompt})
            messages.extend(process.state)

            tokens = num_tokens_from_messages(messages, model=process.model_name)
            window_size = get_context_window_size(process.model_name, CONTEXT_WINDOW_SIZES)

            return {
                "input_tokens": tokens,
                "context_window": window_size,
                "percentage": (tokens / window_size * 100) if window_size > 0 else 0,
                "remaining_tokens": max(0, window_size - tokens),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Token counting failed: {exc}")
            return {"error": str(exc)}
