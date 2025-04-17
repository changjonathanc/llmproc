"""Anthropic provider tools implementation for LLMProc."""

import asyncio
import json
import logging
from typing import Any, Optional

# Import Anthropic clients (will be None if not installed)
try:
    from anthropic import AsyncAnthropic, AsyncAnthropicVertex
except ImportError:
    AsyncAnthropic = None
    AsyncAnthropicVertex = None

from llmproc.common.results import RunResult, ToolResult
from llmproc.providers.anthropic_utils import (
    add_cache_to_message,
    add_message_ids,
    add_token_efficient_header_if_needed,
    contains_tool_calls,
    get_context_window_size,
    is_cacheable_content,
    state_to_api_messages,
    system_to_api_format,
    tools_to_api_format,
)
from llmproc.providers.constants import ANTHROPIC_PROVIDERS
from llmproc.providers.utils import safe_callback
from llmproc.utils.message_utils import append_message_with_id

logger = logging.getLogger(__name__)


PROMPT_FORCE_MODEL_RESPONSE = "Please respond with a text response"
PROMPT_SUMMARIZE_CONVERSATION = (
    "Please stop using tools and summarize your progress so far"
)


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
        callbacks: dict = None,
        run_result=None,
        is_tool_continuation: bool = False,
    ) -> "RunResult":
        # PHASE 3: This method contains several complexity hotspots that could be simplified
        # with targeted helper functions, making the code more maintainable without changing structure.
        """Execute a conversation with the Anthropic API.

        This method executes a conversation turn with proper tool handling, tracking metrics,
        and callback notifications. It can be used for both initial user messages and
        for continuing a conversation after tool execution.

        Args:
            process: The LLMProcess instance
            user_prompt: The user's input message
            max_iterations: Maximum number of API calls for tool usage
            callbacks: Optional dictionary of callback functions
            run_result: Optional RunResult object to track execution metrics
            is_tool_continuation: Whether this is continuing a previous tool call

        Returns:
            RunResult object containing execution metrics and API call information
        """
        # Initialize callbacks
        callbacks = callbacks or {}
        on_tool_start = callbacks.get("on_tool_start")
        on_tool_end = callbacks.get("on_tool_end")
        on_response = callbacks.get("on_response")

        if is_tool_continuation:
            pass
        else:
            # Add user message with GOTO ID
            append_message_with_id(process, "user", user_prompt)

        process.run_stop_reason = None
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            goto_executed_this_turn = False  # Flag to track GOTO execution

            logger.debug(f"Making API call {iterations}/{max_iterations}")

            # Make the API call
            api_params = process.api_params.copy()

            # Extract extra headers if present
            extra_headers = (
                api_params.pop("extra_headers", {})
                if "extra_headers" in api_params
                else {}
            )

            # Determine if we should use caching
            # Prompt caching is implemented via cache_control parameters in content
            # This works for both direct Anthropic API and Vertex AI as confirmed by testing
            use_caching = not getattr(process, "disable_automatic_caching", False)

            # Apply token-efficient tool use if appropriate (for Claude 3.7+ on both direct Anthropic API and Vertex AI)
            # Testing confirmed it works on both providers
            # Add token-efficient tools header if appropriate
            extra_headers = add_token_efficient_header_if_needed(process, extra_headers)

            # Warning if token-efficient tools header is present but not supported
            if (
                "anthropic-beta" in extra_headers
                and "token-efficient-tools" in extra_headers["anthropic-beta"]
                and (
                    process.provider not in ANTHROPIC_PROVIDERS
                    or not process.model_name.startswith("claude-3-7")
                )
            ):
                logger.warning(
                    f"Token-efficient tools header is only supported by Claude 3.7 models. Currently using {process.model_name} on {process.provider}. The header will be ignored."
                )

            # Transform internal state to API-ready format with caching
            api_messages = state_to_api_messages(process.state, add_cache=use_caching)
            api_system = system_to_api_format(
                process.enriched_system_prompt, add_cache=use_caching
            )
            api_tools = tools_to_api_format(process.tools, add_cache=use_caching)

            # Make the API call with any extra headers
            response = await process.client.messages.create(
                model=process.model_name,
                system=api_system,
                messages=api_messages,
                tools=api_tools,
                extra_headers=extra_headers if extra_headers else None,
                **api_params,
            )

            # Track API call in the run result if available
            if run_result:
                # Extract API usage information
                api_info = {
                    "model": process.model_name,
                    "usage": getattr(response, "usage", {}),
                    "stop_reason": getattr(response, "stop_reason", None),
                    "id": getattr(response, "id", None),
                }
                run_result.add_api_call(api_info)

            stop_reason = response.stop_reason

            has_tool_calls = contains_tool_calls(response.content)
            tool_results = []
            # NOTE: these are the possible stop_reason values: ["end_turn", "max_tokens", "stop_sequence"]:
            process.stop_reason = stop_reason  # TODO: not finalized api,
            if not has_tool_calls:
                if response.content:
                    # NOTE: sometimes model can decide to not response any text, for example, after using tools.
                    # appending the empty assistant message will cause the following API error in the next api call:
                    # ERROR: all messages must have non-empty content except for the optional final assistant message
                    append_message_with_id(process, "assistant", response.content)
                # NOTE: this is needed for user to check the stop reason afterward
                process.run_stop_reason = stop_reason
                break
            else:
                # PHASE 3: Text content extraction could be simplified with a helper function
                # _extract_text_content(content_items) to reduce code duplication and improve readability.

                # Fire callback for model response if provided
                if on_response and "on_response" in callbacks:
                    # Extract text content for callback
                    text_content = ""
                    for c in response.content:
                        if (
                            hasattr(c, "type")
                            and c.type == "text"
                            and hasattr(c, "text")
                        ):
                            text_content += c.text
                    safe_callback(
                        on_response, text_content, callback_name="on_response"
                    )

                for content in response.content:
                    if content.type == "text":
                        continue
                        # NOTE: right now the text response will be appended to messages list later
                    elif content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_id = content.id

                        # Fire callback for tool start if provided
                        safe_callback(
                            on_tool_start,
                            tool_name,
                            tool_args,
                            callback_name="on_tool_start",
                        )

                        # Track tool in run_result if available
                        if run_result:
                            run_result.add_tool_call(
                                {
                                    "type": "tool_call",
                                    "tool_name": tool_name,
                                    "args": tool_args,
                                }
                            )

                        # PHASE 3: The tool execution conditional could be simplified with a
                        # _is_fork_tool(tool_name) helper function for better readability.

                        # NOTE: fork requires special handling, such as removing all other tool calls from the last assistant response
                        # so we separate the fork handling from other tool call handling
                        if tool_name == "fork":
                            logger.info(f"Forking with tool_args: {tool_args}")
                            result = await self._fork(
                                process,
                                tool_args,
                                tool_id,
                                last_assistant_response=response.content,
                            )
                        else:
                            # Handle tool call with parameters
                            if isinstance(tool_args, dict):
                                # Call the tool with explicit keyword arguments extracted from the input dict
                                # process.call_tool already handles exceptions internally and returns a ToolResult
                                logger.debug(
                                    f"Calling tool '{tool_name}' with parameters: {tool_args}"
                                )
                                result = await process.call_tool(tool_name, **tool_args)
                            else:
                                # This case shouldn't happen with properly formatted API responses
                                error_msg = f"Invalid tool arguments format for '{tool_name}'. Expected a dictionary but got: {type(tool_args)}"
                                logger.error(error_msg)
                                result = ToolResult.from_error(error_msg)
                                
                        # Fire callback for tool end if provided
                        safe_callback(
                            on_tool_end, tool_name, result, callback_name="on_tool_end"
                        )

                        # Check if GOTO was just executed and set flag
                        if tool_name == "goto":
                            logger.info(
                                "GOTO tool executed. Setting flag to skip appending messages for this tool response."
                            )
                            goto_executed_this_turn = True
                            break  # Exit the loop processing tools for this API response

                        # Check if GOTO was just executed and set flag
                        if tool_name == "goto":
                            logger.info(f"GOTO tool executed. Setting flag to skip appending messages for this tool response.")
                            goto_executed_this_turn = True
                            break  # Exit the loop processing tools for this API response

                        # Check if GOTO was just executed and set flag
                        if tool_name == "goto":
                            logger.info(
                                "GOTO tool executed. Setting flag to skip appending messages for this tool response."
                            )
                            goto_executed_this_turn = True
                            break  # Exit the loop processing tools for this API response

                        # Check if GOTO was just executed and set flag
                        if tool_name == "goto":
                            logger.info(
                                "GOTO tool executed. Setting flag to skip appending messages for this tool response."
                            )
                            goto_executed_this_turn = True
                            break  # Exit the loop processing tools for this API response

                        # PHASE 3: This file descriptor eligibility check is complex and could be simplified with a
                        # _should_use_file_descriptor(process, tool_name, tool_result) helper function.

                        # Require a ToolResult instance
                        if not isinstance(result, ToolResult):
                            # This is a programming error - tools must return ToolResult
                            error_msg = f"Tool '{tool_name}' did not return a ToolResult instance. Got {type(result).__name__} instead."
                            logger.error(error_msg)
                            # Create an error ToolResult to avoid breaking the conversation
                            tool_result = ToolResult.from_error(error_msg)
                        else:
                            tool_result = result

                            # Check if file descriptor should be used for this tool result
                            if (
                                hasattr(process, "file_descriptor_enabled")
                                and process.file_descriptor_enabled
                                and hasattr(process, "fd_manager")
                                and process.fd_manager
                            ):
                                # Use helper to decide if FD is needed and create it if so
                                processed_result, used_fd = process.fd_manager.create_fd_from_tool_result(
                                    tool_result.content, tool_name
                                )

                                if used_fd:
                                    logger.info(
                                        f"Tool result from '{tool_name}' exceeds {process.fd_manager.max_direct_output_chars} chars, creating file descriptor"
                                    )
                                    tool_result = processed_result
                                    logger.debug(
                                        f"Created file descriptor for tool result from '{tool_name}'"
                                    )

                        # Only convert to dict at the last moment when building the response
                        tool_result_dict = tool_result.to_dict()
                        # TODO: maybe tool_result should support to_anthropic and other providers format

                        # Append to tool results
                        tool_results.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_id,
                                        **tool_result_dict,
                                    }
                                ],
                            }
                        )
                # PHASE 3: The state update could be made more intentional with a
                # _update_state_with_tool_results(process, response, tool_results) helper function.

                # Only update state if GOTO was NOT executed in this turn
                if not goto_executed_this_turn:
                    # Update state with response and tool results
                    # Add assistant message with GOTO ID
                    append_message_with_id(process, "assistant", response.content)

                    # Add tool results to state using append_message_with_id
                    for tool_result in tool_results:
                        # Tool result already has the correct structure with "role": "user"
                        # and "content" is a list with tool result data
                        append_message_with_id(process, tool_result["role"], tool_result["content"])
                else:
                    logger.debug("GOTO executed this turn, skipping state update for assistant/tool results.")

        if iterations >= max_iterations:
            process.run_stop_reason = "max_iterations"

        # Create a new RunResult if one wasn't provided
        if run_result is None:
            run_result = RunResult()
            run_result.api_calls = iterations

        # Complete the RunResult and return it
        return run_result.complete()

    async def run_till_text_response(
        self, process, user_prompt, max_iterations: int = 10
    ) -> str:
        """Run the process until a text response is generated.

        This is specifically designed for forked processes, where the child must respond with a text response, which will become the tool result for the parent.

        This has some special handling, it's not meant for general use.

        Args:
            process: The LLMProcess instance
            user_prompt: The user's input message
            max_iterations: Maximum number of API calls for tool usage

        Returns:
            The text response from the model, or an error message if no response is generated
        """
        # Create a combined RunResult to track all API calls
        master_run_result = RunResult()
        next_prompt = user_prompt
        callbacks = {}  # No callbacks for this internal method

        while master_run_result.api_calls < max_iterations:
            # Run the process and get a RunResult
            await self.run(
                process,
                next_prompt,
                max_iterations=max_iterations - master_run_result.api_calls,
                callbacks=callbacks,
                run_result=master_run_result,  # Pass the master RunResult to accumulate metrics
                is_tool_continuation=False,
            )

            # Check if we've reached the text response we need
            last_message = process.get_last_message()
            if last_message:
                # Complete the RunResult before returning
                master_run_result.complete()
                return last_message  # Return the text message directly

            # If we didn't get a response, handle special cases
            if process.run_stop_reason == "max_iterations":
                # Allow the model another chance to respond with a text response to summarize
                await self.run(
                    process,
                    PROMPT_SUMMARIZE_CONVERSATION,
                    max_iterations=1,
                    callbacks=callbacks,
                    run_result=master_run_result,  # Pass the master RunResult
                    is_tool_continuation=False,
                )

                # Check again for a text response
                last_message = process.get_last_message()
                if last_message:
                    # Complete the RunResult before returning
                    master_run_result.complete()
                    return last_message

            # If we still don't have a text response, prompt again
            next_prompt = PROMPT_FORCE_MODEL_RESPONSE

        # If we've exhausted iterations without getting a proper response
        master_run_result.complete()
        return "Maximum iterations reached without final response."

    async def count_tokens(self, process) -> dict:
        """Count tokens in the current conversation context using Anthropic's API.

        Args:
            process: The LLMProcess instance

        Returns:
            dict: Token count information or error message
        """
        try:
            # Create a copy of the state and always append a dummy user message
            # This prevents API errors with trailing whitespace in final assistant messages
            state_copy = (process.state or []).copy()
            state_copy.append({"role": "user", "content": "Hi"})
            api_messages = state_to_api_messages(state_copy, add_cache=False)

            # Handle system prompt format
            system_prompt = process.enriched_system_prompt

            # Get tool definitions if available
            api_tools = (
                tools_to_api_format(process.tools, add_cache=False)
                if hasattr(process, "tools")
                else None
            )

            # Call Anthropic's count_tokens API
            params = {
                "model": process.model_name,
                "messages": api_messages,
            }

            if system_prompt:
                params["system"] = system_prompt

            if api_tools:
                params["tools"] = api_tools

            # Use the messages.count_tokens endpoint
            response = await process.client.messages.count_tokens(**params)

            # Calculate context window percentage
            window_size = get_context_window_size(
                process.model_name, self.CONTEXT_WINDOW_SIZES
            )
            input_tokens = getattr(response, "input_tokens", 0)
            percentage = (input_tokens / window_size * 100) if window_size > 0 else 0
            remaining = max(0, window_size - input_tokens)

            return {
                "input_tokens": input_tokens,
                "context_window": window_size,
                "percentage": percentage,
                "remaining_tokens": remaining,
            }

        except Exception as e:
            logger.warning(f"Token counting failed: {str(e)}")
            return {"error": str(e)}

    #
    # Tool handling methods
    # PHASE 3: This section would be enhanced with targeted helper functions that simplify complexity hotspots:
    # - _extract_text_content(content_items)
    # - _contains_tool_calls(response_content)
    # - _safe_callback(callback_fn, *args, callback_name)
    # - _should_use_file_descriptor(process, tool_name, tool_result)
    # - _add_token_efficient_header_if_needed(process, headers)
    #

    @staticmethod
    async def _fork(process, params, tool_id, last_assistant_response):
        """Fork a conversation."""
        # PHASE 3: Error handling here could be improved with _safe_callback pattern
        # and better validation of input parameters
        if not process.allow_fork:
            return ToolResult.from_error(
                "Forking is not allowed for this agent, possible reason: You are already a forked instance"
            )

        prompts = params["prompts"]
        logger.info(f"Forking conversation with {len(prompts)} prompts: {prompts}")

        async def process_fork(i, prompt):
            # Use the fork_process method to create a deep copy
            child = await process.fork_process()

            # NOTE: we need to remove all other tool calls from the last assistant response
            # because we might not have the tool call results for other tool calls yet
            # this is also important for the forked process to focus on the assigned goal
            child.state.append(
                {
                    "role": "assistant",
                    "content": [
                        content
                        for content in last_assistant_response
                        if content.type != "tool_use" or (content.type == "tool_use" and content.id == tool_id)
                    ],
                }
            )
            # NOTE: return the fork result as tool result
            child.state.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": "pid==0, you are a child instance produced from a fork. you are not allowed to use the fork tool. please continue the conversation with only the assigned goal",
                        }
                    ],
                }
            )
            # NOTE: run() will immediately add the prompt to the conversation as user message
            # I found this to work better than adding the prompt as the tool result
            executor = AnthropicProcessExecutor()  # Create a new executor for the child
            response = await executor.run_till_text_response(
                child, user_prompt=prompt, max_iterations=20
            )
            return {"id": i, "message": response}

        # Process all forks in parallel
        responses = await asyncio.gather(
            *[process_fork(i, prompt) for i, prompt in enumerate(prompts)]
        )

        # Return results as a ToolResult object
        return ToolResult.from_success(json.dumps(responses))
