"""Anthropic provider tools implementation for LLMProc."""

import asyncio
import copy
import json
import logging

# Import Anthropic clients (will be None if not installed)
try:
    from anthropic import AsyncAnthropic, AsyncAnthropicVertex
except ImportError:
    AsyncAnthropic = None
    AsyncAnthropicVertex = None

from llmproc.providers.constants import ANTHROPIC_PROVIDERS
from llmproc.results import RunResult
from llmproc.tools.tool_result import ToolResult

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
    async def run(
        self,
        process: "Process",  # noqa: F821
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
            api_params = process.api_params.copy()
            
            # Extract extra headers if present
            extra_headers = api_params.pop("extra_headers", {})
            
            # Determine if we should use caching
            # Prompt caching is implemented via cache_control parameters in content
            # This works for both direct Anthropic API and Vertex AI as confirmed by testing
            use_caching = not getattr(process, "disable_automatic_caching", False)
            
            # Apply token-efficient tool use if appropriate (for Claude 3.7+ on both direct Anthropic API and Vertex AI)
            # Testing confirmed it works on both providers
            if process.provider in ANTHROPIC_PROVIDERS and process.model_name.startswith("claude-3-7"):
                # Add token-efficient tools beta header if appropriate
                if "anthropic-beta" not in extra_headers:
                    extra_headers["anthropic-beta"] = "token-efficient-tools-2025-02-19"
                elif "token-efficient-tools" not in extra_headers["anthropic-beta"]:
                    # Append to existing beta features
                    extra_headers["anthropic-beta"] += ",token-efficient-tools-2025-02-19"
            elif ("anthropic-beta" in extra_headers and 
                  "token-efficient-tools" in extra_headers["anthropic-beta"] and
                  (process.provider not in ANTHROPIC_PROVIDERS or not process.model_name.startswith("claude-3-7"))):
                # Warning if token-efficient tools header is present but not supported
                logger.warning(
                    f"Token-efficient tools header is only supported by Claude 3.7 models. "
                    f"Currently using {process.model_name} on {process.provider}. The header will be ignored."
                )
            
            # Transform internal state to API-ready format with caching
            api_messages = self._state_to_api_messages(process.state, add_cache=use_caching)
            api_system = self._system_to_api_format(process.enriched_system_prompt, add_cache=use_caching)
            api_tools = self._tools_to_api_format(process.tools, add_cache=use_caching)
            
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
                            run_result.add_tool_call(
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
                            
                            # Check if file descriptor system is enabled and output exceeds the threshold
                            if (hasattr(process, "file_descriptor_enabled") and 
                                process.file_descriptor_enabled and 
                                hasattr(process, "fd_manager") and 
                                process.fd_manager and 
                                not process.fd_manager.is_fd_related_tool(tool_name) and  # Skip FD-related tools
                                isinstance(tool_result.content, str) and 
                                len(tool_result.content) > process.fd_manager.max_direct_output_chars):
                                
                                logger.info(f"Tool result from '{tool_name}' exceeds {process.fd_manager.max_direct_output_chars} chars, creating file descriptor")
                                
                                # Create a file descriptor for the large content
                                fd_result = process.fd_manager.create_fd(tool_result.content)
                                
                                # Replace the original tool result with the file descriptor result
                                # fd_result is already a ToolResult instance now
                                tool_result = fd_result
                                
                                logger.debug(f"Created file descriptor for tool result from '{tool_name}'")

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

    def _state_to_api_messages(self, state, add_cache=True):
        """
        Transform conversation state to API-ready messages, optionally adding cache control points.

        Args:
            state: The conversation state to transform
            add_cache: Whether to add cache control points

        Returns:
            List of API-ready messages with cache_control added at strategic points
        """
        # Create a deep copy to avoid modifying the original state
        messages = copy.deepcopy(state)

        if not add_cache or not messages:
            return messages

        # Add cache to the last message regardless of type
        if messages:
            self._add_cache_to_message(messages[-1])

        # Find non-tool user messages
        non_tool_user_indices = []
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                # Check if this is not a tool result message
                is_tool_message = False
                if isinstance(msg.get("content"), list):
                    for content in msg["content"]:
                        if isinstance(content, dict) and content.get("type") == "tool_result":
                            is_tool_message = True
                            break
                
                if not is_tool_message:
                    non_tool_user_indices.append(i)
        
        # Add cache to the message before the most recent non-tool user message
        if len(non_tool_user_indices) > 1:
            before_last_user_index = non_tool_user_indices[-2]
            if before_last_user_index > 0:  # Ensure there's a message before this one
                self._add_cache_to_message(messages[before_last_user_index - 1])

        return messages
    
    def _system_to_api_format(self, system_prompt, add_cache=True):
        """
        Transform system prompt to API-ready format with cache control.

        Args:
            system_prompt: The enriched system prompt
            add_cache: Whether to add cache control

        Returns:
            API-ready system prompt with cache_control
        """
        if not add_cache:
            return system_prompt

        if isinstance(system_prompt, str):
            # Add cache to the entire system prompt
            return [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        elif isinstance(system_prompt, list):
            # Already in structured format, assume correctly configured
            return system_prompt
        else:
            # Fallback for unexpected formats
            return system_prompt
    
    def _tools_to_api_format(self, tools, add_cache=True):
        """
        Transform tools to API-ready format with cache control.
        
        Args:
            tools: The tool definitions
            add_cache: Whether to add cache control
            
        Returns:
            API-ready tools with cache_control
        """
        if not add_cache or not tools:
            return tools
            
        tools_copy = copy.deepcopy(tools)
        
        # Add cache_control to the last tool in the array
        if isinstance(tools_copy, list) and tools_copy:
            # Find the last tool and add cache_control to it
            # This caches all tools up to this point, using just one cache point
            if isinstance(tools_copy[-1], dict):
                tools_copy[-1]["cache_control"] = {"type": "ephemeral"}
                
        return tools_copy
    
    def _add_cache_to_message(self, message):
        """Add cache control to a message."""
        if isinstance(message.get("content"), list):
            for content in message["content"]:
                if isinstance(content, dict) and content.get("type") in ["text", "tool_result"]:
                    content["cache_control"] = {"type": "ephemeral"}
                    return  # Only add to the first eligible content
        elif isinstance(message.get("content"), str):
            # Convert string content to structured format with cache
            message["content"] = [{"type": "text", "text": message["content"], "cache_control": {"type": "ephemeral"}}]
    
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
