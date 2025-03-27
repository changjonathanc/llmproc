"""OpenAI provider implementation for LLMProc."""

import logging

from llmproc.results import RunResult

logger = logging.getLogger(__name__)


class OpenAIProcessExecutor:
    """Process executor for OpenAI models.

    This is a simplified version that doesn't support tools yet.
    Tool support will be added in future versions.
    """

    async def run(
        self,
        process: "Process",  # noqa: F821
        user_prompt: str,
        max_iterations: int = 1,
        callbacks: dict = None,
        run_result=None,
        is_tool_continuation: bool = False,
    ) -> "RunResult":
        """Execute a conversation with the OpenAI API.

        Args:
            process: The LLMProcess instance
            user_prompt: The user's input message
            max_iterations: Not used in OpenAI executor as tools aren't supported yet
            callbacks: Optional dictionary of callback functions
            run_result: Optional RunResult object to track execution metrics
            is_tool_continuation: Not used in OpenAI executor as tools aren't supported yet

        Returns:
            RunResult object containing execution metrics and API call information

        Raises:
            ValueError: If tools are configured but not yet supported
        """
        # Initialize callbacks
        callbacks = callbacks or {}
        on_response = callbacks.get("on_response")

        # Check if tools are configured but not yet supported
        if process.tools and len(process.tools) > 0:
            raise ValueError(
                "Tool usage is not yet supported for OpenAI models in this implementation. "
                "Please use a model without tools, or use the Anthropic provider for tool support."
            )

        # Add user message to conversation history
        process.state.append({"role": "user", "content": user_prompt})

        # Set up messages for OpenAI format
        formatted_messages = []

        # First add system message if present
        if process.enriched_system_prompt:
            formatted_messages.append(
                {"role": "system", "content": process.enriched_system_prompt}
            )

        # Then add conversation history
        for message in process.state:
            # Add user and assistant messages
            if message["role"] in ["user", "assistant"]:
                formatted_messages.append(
                    {"role": message["role"], "content": message["content"]}
                )

        # Create a new RunResult if one wasn't provided
        if run_result is None:
            run_result = RunResult()

        logger.debug(f"Making OpenAI API call with {len(formatted_messages)} messages")

        try:
            # Make the API call
            # Check if this is a reasoning model (o1, o1-mini, o3, o3-mini)
            api_params = process.api_params.copy()
            
            # Only pass reasoning_effort for reasoning models
            is_reasoning_model = process.model_name.startswith(("o1", "o3"))
            if not is_reasoning_model and "reasoning_effort" in api_params:
                del api_params["reasoning_effort"]
                
            response = await process.client.chat.completions.create(
                model=process.model_name,
                messages=formatted_messages,
                **api_params,
            )

            # Track API call in the run result
            api_info = {
                "model": process.model_name,
                "usage": getattr(response, "usage", {}),
                "id": getattr(response, "id", None),
            }
            run_result.add_api_call(api_info)

            # Extract the response message content
            message_content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Set stop reason
            process.run_stop_reason = finish_reason

            # Add assistant response to conversation history
            process.state.append({"role": "assistant", "content": message_content})

            # Fire callback for model response if provided
            if on_response:
                try:
                    on_response(message_content)
                except Exception as e:
                    logger.warning(f"Error in on_response callback: {str(e)}")

        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            # Add error to run result
            run_result.add_api_call({"type": "error", "error": str(e)})
            process.run_stop_reason = "error"
            raise

        # Complete the RunResult and return it
        return run_result.complete()

    # TODO: Implement tool support
    # TODO: Implement run_till_text_response method for forked processes
    # TODO: Implement _fork method for conversation forking
