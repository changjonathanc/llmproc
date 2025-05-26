"""OpenAI provider implementation for LLMProc.

NOTE: This implementation is minimally maintained as we plan to integrate with LiteLLM
in a future release for more comprehensive provider support once Anthropic and core
functionality are mature enough.
"""

import logging

from llmproc.callbacks import CallbackEvent
from llmproc.common.results import RunResult

logger = logging.getLogger(__name__)


class OpenAIProcessExecutor:
    """Process executor for OpenAI models.

    This is a simplified version that doesn't support tools yet.
    Tool support will be added in future versions.

    Note: This executor is minimally maintained as we plan to replace provider-specific
    executors with LiteLLM in a future release for unified provider support.
    """

    async def run(
        self,
        process: "Process",  # noqa: F821
        user_prompt: str,
        max_iterations: int = 1,
        run_result=None,
        is_tool_continuation: bool = False,
    ) -> "RunResult":
        """Execute a conversation with the OpenAI API.

        Args:
            process: The LLMProcess instance
            user_prompt: The user's input message
            max_iterations: Not used in OpenAI executor as tools aren't supported yet
            run_result: Optional RunResult object to track execution metrics
            is_tool_continuation: Not used in OpenAI executor as tools aren't supported yet

        Returns:
            RunResult object containing execution metrics and API call information

        Raises:
            ValueError: If tools are configured but not yet supported
        """
        # Prepare for response handling

        # Check if tools are configured but not yet supported
        if process.tools and len(process.tools) > 0:
            raise ValueError(
                "Tool usage is not yet supported for OpenAI models in this implementation. Please use a model without tools, or use the Anthropic provider for tool support."
            )

        # Add user message to conversation history
        process.state.append({"role": "user", "content": user_prompt})

        # Trigger TURN_START event
        process.trigger_event(CallbackEvent.TURN_START, process)

        # Set up messages for OpenAI format
        formatted_messages = []

        # First add system message if present
        if process.enriched_system_prompt:
            formatted_messages.append({"role": "system", "content": process.enriched_system_prompt})

        # Then add conversation history
        for message in process.state:
            # Add user and assistant messages
            if message["role"] in ["user", "assistant"]:
                formatted_messages.append({"role": message["role"], "content": message["content"]})

        # Create a new RunResult if one wasn't provided
        if run_result is None:
            run_result = RunResult()

        logger.debug(f"Making OpenAI API call with {len(formatted_messages)} messages")

        try:
            # Make the API call
            # Check if this is a reasoning model (o1, o1-mini, o3, o3-mini)
            api_params = process.api_params.copy()

            # Determine if this is a reasoning model
            is_reasoning_model = process.model_name.startswith(("o1", "o3"))

            # Handle reasoning model specific parameters
            if is_reasoning_model:
                # Reasoning models use max_completion_tokens instead of max_tokens
                if "max_tokens" in api_params:
                    api_params["max_completion_tokens"] = api_params.pop("max_tokens")
            else:
                # Remove reasoning_effort for non-reasoning models
                if "reasoning_effort" in api_params:
                    del api_params["reasoning_effort"]

            # Build API request payload
            api_request = {
                "model": process.model_name,
                "messages": formatted_messages,
                "params": api_params,
            }

            # Trigger API request event
            process.trigger_event(CallbackEvent.API_REQUEST, api_request)

            # Make API call

            response = await process.client.chat.completions.create(
                model=process.model_name,
                messages=formatted_messages,
                **api_params,
            )

            # Trigger API response event
            process.trigger_event(CallbackEvent.API_RESPONSE, response)

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

            # Extract the response message content
            message_content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Set stop reason
            process.run_stop_reason = finish_reason

            # Add assistant response to conversation history
            process.state.append({"role": "assistant", "content": message_content})

            # Trigger response event
            if message_content:
                process.trigger_event(CallbackEvent.RESPONSE, message_content)

            # Trigger TURN_END event
            process.trigger_event(CallbackEvent.TURN_END, process, response, [])

        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            # Add error to run result
            run_result.add_api_call({"type": "error", "error": str(e)})
            process.run_stop_reason = "error"
            raise

        # Set the last_message in the RunResult to ensure it's available
        # This is critical for the sync interface tests
        last_message = process.get_last_message()
        run_result.set_last_message(last_message)

        # Complete the RunResult and return it
        return run_result.complete()

    # TODO: Implement tool support
