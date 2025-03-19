"""Anthropic provider tools implementation for LLMProc."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import Anthropic clients (will be None if not installed)
try:
    from anthropic import AsyncAnthropic, AsyncAnthropicVertex
except ImportError:
    AsyncAnthropic = None
    AsyncAnthropicVertex = None

from llmproc.results import RunResult
from llmproc.tools.tool_result import ToolResult

logger = logging.getLogger(__name__)


PROMPT_FORCE_MODEL_RESPONSE = "Please respond with a text response"
PROMPT_SUMMARIZE_CONVERSATION = (
    "Please stop using tools and summarize your progress so far"
)


class AnthropicProcessExecutor:
    async def run(
        self,
        process: "Process",
        user_prompt: str,
        max_iterations: int = 10,
        callbacks: dict = None,
        run_result=None,
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
            process.state.append({"role": "user", "content": user_prompt})

        process.run_stop_reason = None
        iterations = 0
        while iterations < max_iterations:
            iterations += 1

            logger.debug(f"Making API call {iterations}/{max_iterations}")

            # Make the API call
            response = await process.client.messages.create(
                model=process.model_name,
                system=process.enriched_system_prompt,
                messages=process.state,
                tools=process.tools,
                **process.api_params,
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

            has_tool_calls = (
                len(
                    [
                        content
                        for content in response.content
                        if content.type == "tool_use"
                    ]
                )
                > 0
            )
            tool_results = []
            # NOTE: these are the possible stop_reason values: ["end_turn", "max_tokens", "stop_sequence"]:
            process.stop_reason = stop_reason  # TODO: not finalized api,
            if not has_tool_calls:
                if response.content:
                    # NOTE: sometimes model can decide to not response any text, for example, after using tools.
                    # appending the empty assistant message will cause the following API error in the next api call:
                    # ERROR: all messages must have non-empty content except for the optional final assistant message
                    process.state.append(
                        {"role": "assistant", "content": response.content}
                    )
                # NOTE: this is needed for user to check the stop reason afterward
                process.run_stop_reason = stop_reason
                break
            else:
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
                    try:
                        on_response(text_content)
                    except Exception as e:
                        logger.warning(f"Error in on_response callback: {str(e)}")

                for content in response.content:
                    if content.type == "text":
                        continue
                        # NOTE: right now the text response will be appended to messages list later
                    elif content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_id = content.id

                        # Fire callback for tool start if provided
                        if on_tool_start:
                            try:
                                on_tool_start(tool_name, tool_args)
                            except Exception as e:
                                logger.warning(
                                    f"Error in on_tool_start callback: {str(e)}"
                                )

                        # Track tool in run_result if available
                        if run_result:
                            run_result.add_api_call(
                                {
                                    "type": "tool_call",
                                    "tool_name": tool_name,
                                }
                            )

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
                            result = await process.call_tool(tool_name, tool_args)

                        # Fire callback for tool end if provided
                        if on_tool_end:
                            try:
                                on_tool_end(tool_name, result)
                            except Exception as e:
                                logger.warning(
                                    f"Error in on_tool_end callback: {str(e)}"
                                )

                        # Require a ToolResult instance
                        if not isinstance(result, ToolResult):
                            # This is a programming error - tools must return ToolResult
                            error_msg = f"Tool '{tool_name}' did not return a ToolResult instance. Got {type(result).__name__} instead."
                            logger.error(error_msg)
                            # Create an error ToolResult to avoid breaking the conversation
                            tool_result = ToolResult.from_error(error_msg)
                        else:
                            tool_result = result

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
                process.state.append(
                    {"role": "assistant", "content": response.content}
                )
                process.state.extend(tool_results)

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
    ):
        """
        Run the process until a text response is generated
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

    @staticmethod
    async def _fork(process, params, tool_id, last_assistant_response):
        """Fork a conversation."""
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
                        if content.type != "tool_use"
                        or (content.type == "tool_use" and content.id == tool_id)
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
