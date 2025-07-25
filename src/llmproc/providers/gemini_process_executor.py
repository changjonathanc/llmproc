"""Gemini provider implementation for LLMProc using google-genai SDK.

NOTE: This implementation is minimally maintained as we plan to integrate with LiteLLM
in a future release for more comprehensive provider support once Anthropic and core
functionality are mature enough.
"""

import logging

# Import Google Genai SDK (will be None if not installed)
try:
    from google import genai
except ImportError:
    genai = None

from llmproc.callbacks import CallbackEvent
from llmproc.common.results import RunResult
from llmproc.providers.gemini_utils import convert_tools_to_gemini_format, format_tool_result_for_gemini
from llmproc.utils.message_utils import append_message

logger = logging.getLogger(__name__)


class GeminiProcessExecutor:
    """Process executor for Google Gemini models.

    This class manages interactions with the Gemini API via the google-genai SDK,
    including handling basic conversation flow. Works with both direct API and Vertex AI access paths.

    Note: This executor is minimally maintained as we plan to replace provider-specific
    executors with LiteLLM in a future release for unified provider support.
    """

    # Map of model names to context window sizes
    CONTEXT_WINDOW_SIZES = {
        "gemini-1.5-flash": 128000,
        "gemini-1.5-pro": 1000000,
        "gemini-2.0-flash": 128000,
        # gemini-2.0-pro is no longer available
        "gemini-2.5-pro": 1000000,
    }

    def _supports_token_counting(self, client):
        """Check if client supports token counting.

        Args:
            client: The client instance to check

        Returns:
            bool: True if token counting is supported
        """
        return client is not None and hasattr(client, "models") and hasattr(client.models, "count_tokens")

    def _get_estimated_token_count(self, model_name):
        """Return estimation when token counting is not available.

        Args:
            model_name: The model name to get context window size for

        Returns:
            dict: Estimated token count information
        """
        window_size = self._get_context_window_size(model_name)
        return {
            "input_tokens": -1,  # Indicates estimation
            "context_window": window_size,
            "percentage": -1,
            "remaining_tokens": window_size,
            "note": "Token counting not supported by this client, using placeholders",
        }

    def _calculate_window_usage(self, token_count, window_size, cached_count=0):
        """Calculate context window usage from token count values.

        Args:
            token_count: The total number of tokens
            window_size: The context window size
            cached_count: Optional count of cached tokens

        Returns:
            dict: Token usage information
        """
        percentage = (token_count / window_size * 100) if window_size > 0 else 0
        remaining = max(0, window_size - token_count)

        return {
            "input_tokens": token_count,
            "context_window": window_size,
            "percentage": percentage,
            "remaining_tokens": remaining,
            "cached_tokens": cached_count or 0,
        }

    async def run(
        self,
        process: "Process",  # noqa: F821
        user_prompt: str | None = None,
        max_iterations: int = 10,
        is_tool_continuation: bool = False,
    ) -> "RunResult":
        """Execute a conversation with the Gemini API.

        Args:
            process: The LLMProcess instance
            user_prompt: The user's input message (required unless continuing a tool call)
            max_iterations: Maximum number of API calls for tool usage
            is_tool_continuation: Whether this is continuing a previous tool call

        Returns:
            RunResult object containing execution metrics and API call information
        """
        run_result = RunResult()
        iterations = 0

        # Add user message to conversation state if not continuing from a tool call
        if not is_tool_continuation:
            if user_prompt is None:
                raise ValueError("user_prompt is required when not continuing a tool call")
            append_message(process, "user", user_prompt)

        while iterations < max_iterations:
            # Trigger TURN_START event
            await process.trigger_event(CallbackEvent.TURN_START, run_result=run_result)

            # Prepare tools for API call
            formatted_tools = convert_tools_to_gemini_format(process.tools)

            # Prepare messages for the API - convert internal state format to Gemini format
            contents = self.format_state_to_api_messages(process.state)

            # Prepare API parameters
            api_params = self._prepare_api_params(process.api_params)

            # Prepare API call
            api_request = {
                "model": process.model_name,
                "contents": contents,
                "system_instruction": process.enriched_system_prompt,
                "config": api_params,
            }

            # Add tools to API request if available
            if formatted_tools:
                api_request["tools"] = formatted_tools
                api_request["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}

            # Trigger API request event
            await process.trigger_event(CallbackEvent.API_REQUEST, api_request=api_request)

            # Make the API call
            response = await self._make_api_call(
                client=process.client,
                model=process.model_name,
                contents=contents,
                system_instruction=process.enriched_system_prompt,
                config=api_params,
                tools=formatted_tools,
                tool_config={"function_calling_config": {"mode": "AUTO"}} if formatted_tools else None,
            )

            # Trigger API response event
            await process.trigger_event(CallbackEvent.API_RESPONSE, response=response)

            # Process API response
            api_info = {
                "model": process.model_name,
                "id": getattr(response, "id", None),
                "request": api_request,
                "response": response,
            }
            run_result.add_api_call(api_info)

            # Check for tool calls in the response
            tool_calls = []
            response_parts = getattr(response.candidates[0].content, "parts", [])
            for part in response_parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool_calls.append(part.function_call)

            if tool_calls:
                # Handle tool calls
                run_result.set_stop_reason("tool_use")

                # Add assistant message with tool calls to state
                append_message(process, "assistant", [{"tool_calls": tool_calls}])

                tool_results = []
                for tool_call in tool_calls:
                    # Trigger tool call event
                    await process.trigger_event(
                        CallbackEvent.TOOL_START, tool_name=tool_call.name, tool_args=tool_call.args
                    )
                    run_result.add_tool_call(tool_name=tool_call.name, tool_args=tool_call.args)

                    # Execute the tool
                    tool_result = await process.call_tool(tool_call.name, tool_call.args)
                    tool_results.append(tool_result)

                    # Trigger tool result event
                    await process.trigger_event(CallbackEvent.TOOL_END, tool_name=tool_call.name, result=tool_result)

                    # Add tool result to state
                    append_message(process, "tool", tool_result.content)
                    # Store the tool name for proper formatting later
                    process.state[-1]["tool_name"] = tool_call.name

                # Continue the conversation with tool results
                iterations += 1
                continue
            else:
                # Handle text response
                text_response = getattr(response, "text", "")

                # Trigger response event
                if text_response:
                    hook_res = await process.plugins.response(process, text_response)
                    stopped = hook_res is not None and getattr(hook_res, "stop", False)
                    commit = not stopped or getattr(hook_res, "commit_current", True)
                    if commit:
                        append_message(process, "assistant", text_response)
                    if stopped:
                        run_result.set_stop_reason("hook_stop")
                        break
                else:
                    append_message(process, "assistant", text_response)

                # Trigger TURN_END event
                await process.trigger_event(CallbackEvent.TURN_END, response=response, tool_results=[])

                # Set stop reason and break
                run_result.set_stop_reason("end_turn")
                break

        # Set the last_message in the RunResult to ensure it's available
        last_message = process.get_last_message()
        run_result.set_last_message(last_message)

        # Complete the RunResult and return it
        return run_result.complete()

    async def _make_api_call(
        self, client, model, contents, system_instruction=None, config=None, tools=None, tool_config=None
    ):
        """Make a call to the Gemini API using the google-genai SDK.

        Uses the native async API provided by the SDK.

        Args:
            client: The google-genai Client instance
            model: The model name to use
            contents: The content to send (single message or list of messages)
            system_instruction: Optional system instruction
            config: Optional API parameters
            tools: Optional tools in Gemini format
            tool_config: Optional tool configuration

        Returns:
            Response from the Gemini API
        """
        # Create the full configuration for the request
        full_config = {}

        # Add system instruction if provided
        if system_instruction:
            full_config["system_instruction"] = system_instruction

        # Add any other configuration parameters
        if config:
            full_config.update(config)

        # Add tool configuration if tools are provided
        if tools and tool_config:
            full_config.update(tool_config)

        try:
            # Build the API call parameters
            call_params = {
                "model": model,
                "contents": contents,
            }

            # Add tools if provided
            if tools:
                call_params["tools"] = tools

            # Add config if not empty
            if full_config:
                call_params["config"] = full_config

            # Use the native async API provided by the SDK
            return await client.aio.models.generate_content(**call_params)
        except Exception as e:
            # Handle API errors
            error_message = str(e)
            logger.error(f"Error calling Gemini API: {error_message}")

            # Different handling based on error type
            if "rate limit" in error_message.lower():
                # Rate limit error
                raise ValueError(f"Gemini API rate limit exceeded: {error_message}")
            elif "invalid authentication" in error_message.lower() or "permission" in error_message.lower():
                # Authentication error
                raise ValueError(f"Gemini API authentication error: {error_message}")
            else:
                # General API error
                raise ValueError(f"Gemini API error: {error_message}")

    def format_state_to_api_messages(self, state):
        """
        Convert internal state to Gemini API format.

        Following our unified API payload preparation pattern, this function
        formats conversation state into the structure expected by the Gemini API.

        Args:
            state: The conversation state to convert

        Returns:
            List of messages in Gemini API format
        """
        if not state:
            return []

        # Convert conversation state to Gemini format
        messages = []

        for message in state:
            role = message["role"]
            content = message["content"]

            # Process based on role
            if role == "user":
                # Handle user messages
                if isinstance(content, str):
                    # Simple text content
                    messages.append({"role": "user", "parts": [{"text": content}]})
                elif isinstance(content, list):
                    # Complex content (e.g., tool results for future implementation)
                    # For now, just join all content as text
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append({"text": item["text"]})
                        elif isinstance(item, dict) and "content" in item:
                            parts.append({"text": item["content"]})
                        else:
                            parts.append({"text": str(item)})
                    messages.append({"role": "user", "parts": parts})
                else:
                    # Fallback for unknown content types
                    messages.append({"role": "user", "parts": [{"text": str(content)}]})
            elif role == "assistant":
                # Handle assistant messages
                if isinstance(content, str):
                    # Simple text content
                    messages.append({"role": "model", "parts": [{"text": content}]})
                elif isinstance(content, list):
                    # Complex content - check for tool calls
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and "tool_calls" in item:
                            # Handle tool calls
                            for tool_call in item["tool_calls"]:
                                if genai and hasattr(genai.types, "FunctionCall"):
                                    # Use SDK's FunctionCall type
                                    parts.append(genai.types.FunctionCall(name=tool_call.name, args=tool_call.args))
                                else:
                                    # Fallback structure
                                    parts.append({"function_call": {"name": tool_call.name, "args": tool_call.args}})
                        elif isinstance(item, dict) and "text" in item:
                            parts.append({"text": item["text"]})
                        else:
                            parts.append({"text": str(item)})
                    messages.append({"role": "model", "parts": parts})
                else:
                    # Fallback for unknown content types
                    messages.append({"role": "model", "parts": [{"text": str(content)}]})
            elif role == "tool":
                # Handle tool results
                if isinstance(content, str):
                    # Get the tool name from the message metadata
                    tool_name = message.get("tool_name", "unknown_tool")

                    # For Gemini, tool results are formatted as FunctionResponse
                    if genai and hasattr(genai.types, "FunctionResponse"):
                        messages.append(
                            {
                                "role": "function",
                                "parts": [genai.types.FunctionResponse(name=tool_name, response={"result": content})],
                            }
                        )
                    else:
                        # Fallback format
                        messages.append(
                            {
                                "role": "function",
                                "parts": [{"function_response": {"name": tool_name, "response": content}}],
                            }
                        )

        return messages

    def _prepare_api_params(self, api_params):
        """Prepare API parameters for Gemini."""
        if not api_params:
            return {}

        # Make a copy to avoid modifying the original
        params = {}

        # Map common parameters to Gemini parameters
        param_mapping = {
            "temperature": "temperature",
            "max_tokens": "max_output_tokens",
            "top_p": "top_p",
            "top_k": "top_k",
            "stop": "stop_sequences",
        }

        for param_name, param_value in api_params.items():
            # Map parameter names if needed
            if param_name in param_mapping:
                params[param_mapping[param_name]] = param_value
            else:
                # Pass through other parameters
                params[param_name] = param_value

        return params

    async def count_tokens(self, process):
        """Count tokens in the current conversation context using Gemini's API.

        Args:
            process: The LLMProcess instance

        Returns:
            dict: Token count information or error message
        """
        try:
            # Check if client exists and supports token counting
            if not self._supports_token_counting(process.client):
                return self._get_estimated_token_count(process.model_name)

            # Create a text representation of the conversation for token counting
            text_content = ""

            # Add system prompt if present
            if process.enriched_system_prompt:
                text_content += process.enriched_system_prompt + "\n\n"

            # Add conversation history
            for message in process.state:
                if isinstance(message.get("content"), str):
                    text_content += message.get("content", "") + "\n"
                elif isinstance(message.get("content"), list):
                    # Extract text from complex content (for future tool support)
                    text_parts = []
                    for item in message.get("content", []):
                        if isinstance(item, dict):
                            if "text" in item:
                                text_parts.append(item["text"])
                            elif "content" in item:
                                text_parts.append(item["content"])
                        else:
                            text_parts.append(str(item))
                    text_content += " ".join(text_parts) + "\n"

            try:
                # Convert conversation to Gemini's expected format
                contents = self.format_state_to_api_messages(process.state)

                # Add system prompt if present (needs to be formatted for the API)
                if process.enriched_system_prompt:
                    # System instructions are handled differently in the count_tokens API
                    # For accurate counting, include it in the config
                    config = {"system_instruction": process.enriched_system_prompt}
                    token_count_response = process.client.models.count_tokens(
                        model=process.model_name, contents=contents, config=config
                    )
                else:
                    # Call without system instructions
                    token_count_response = process.client.models.count_tokens(
                        model=process.model_name, contents=contents
                    )

                # Get the token count from response
                token_count = getattr(token_count_response, "total_tokens", 0)

                # Check for cached content token count (used in some Gemini models)
                cached_count = getattr(token_count_response, "cached_content_token_count", 0)
                if cached_count:
                    logger.debug(f"Cached content tokens: {cached_count}")

                # Get context window size
                window_size = self._get_context_window_size(process.model_name)

                # Calculate window usage metrics and return
                return self._calculate_window_usage(token_count, window_size, cached_count)
            except Exception as token_error:
                # If token counting fails, log it and return an error
                logger.warning(f"Token counting failed: {str(token_error)}")
                window_size = self._get_context_window_size(process.model_name)
                return {
                    "error": f"Token counting failed: {str(token_error)}",
                    "context_window": window_size,
                }

        except Exception as e:
            return {"error": str(e)}

    def _get_context_window_size(self, model_name):
        """Get the context window size for the given model."""
        # Extract model family
        base_model = model_name

        # Find matching prefix in our size map
        for prefix in self.CONTEXT_WINDOW_SIZES:
            if base_model.startswith(prefix):
                return self.CONTEXT_WINDOW_SIZES[prefix]

        # Default fallback
        return 128000  # Reasonable default for most Gemini models
