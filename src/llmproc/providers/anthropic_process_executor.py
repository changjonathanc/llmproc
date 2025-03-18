"""Anthropic provider tools implementation for LLMProc."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import asyncio
import logging

logger = logging.getLogger(__name__)


PROMPT_FORCE_MODEL_RESPONSE = "Please respond with a text response"
PROMPT_SUMMARIZE_CONVERSATION = "Please stop using tools and summarize your progress so far"

class AnthropicProcessExecutor:
    async def run(self, process: 'Process', user_prompt: str, max_iterations: int = 10, is_tool_continuation: bool = False) -> int:
        """
        Returns:
            int: The number of iterations (api calls) used
        """
        if is_tool_continuation:
            pass
        else:
            process.messages.append({"role": "user", "content": user_prompt})

        process.run_stop_reason = None
        iterations = 0
        while iterations < max_iterations:

            iterations += 1

            response = await process.client.messages.create(
                model=process.model_name,
                system=process.enriched_system_prompt,
                messages=[message for message in process.messages if message["role"] != "system"],
                tools=process.tools,
                **process.api_params
            )

            stop_reason = response.stop_reason

            has_tool_calls = len([content for content in response.content if content.type == "tool_use"]) > 0
            tool_results = []
            # NOTE: these are the possible stop_reason values: ["end_turn", "max_tokens", "stop_sequence"]:
            process.stop_reason = stop_reason # TODO: not finalized api,
            if not has_tool_calls:
                if response.content:
                    # NOTE: sometimes model can decide to not response any text, for example, after using tools.
                    # appending the empty assistant message will cause the following API error in the next api call:
                    # ERROR: all messages must have non-empty content except for the optional final assistant message
                    process.state.append({"role": "assistant", "content": response.content})
                # NOTE: this is needed for user to check the stop reason afterward
                process.run_stop_reason = stop_reason
                break
            else:
                for content in response.content:
                    if content.type == "text":
                        continue
                        # NOTE: right now the text response will be appended to messages list later
                        # TODO: add callback for text response
                    elif content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_id = content.id
                        # TODO: add callback for tool use

                        # NOTE: fork requires special handling, such as removing all other tool calls from the last assistant response
                        # so we separate the fork handling from other tool call handling
                        if tool_name == "fork":
                            logger.info(f"Forking with tool_args: {tool_args}")
                            result = await self._fork(
                                process, tool_args, tool_id, last_assistant_response=response.content
                            )
                        else:
                            result = await process.call_tool(tool_name, tool_args)
                        tool_results.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": result,
                                }
                            ],
                        })
                process.messages.append({"role": "assistant", "content": response.content})
                process.messages.extend(tool_results)
        if iterations >= max_iterations:
            process.run_stop_reason = "max_iterations"
        return iterations

    async def run_till_text_response(self, process, user_prompt, max_iterations: int = 10):
        """
        Run the process until a text response is generated
        This is specifically designed for forked processes, where the child must respond with a text response, which will become the tool result for the parent.

        This has some special handling, it's not meant for general use.
        """

        iterations = 0
        next_prompt = user_prompt
        while iterations < max_iterations:
            iterations += await self.run(process, next_prompt, max_iterations=max_iterations-iterations, is_tool_continuation=False)
            # NOTE: check if the last message is a text response
            # TODO: maybe add a helper method for this

            if process.run_stop_reason == "max_iterations":
                # TODO: we might want to handle this case differently,
                # Currently, we allow the model another chance to respond with a text response to summarize the conversation
                iterations += await self.run(process, PROMPT_SUMMARIZE_CONVERSATION, max_iterations=1, is_tool_continuation=False)

            # now we check if the last message is a text response
            last_message = process.messages[-1]
            if last_message["role"] != "assistant":
                # NOTE: this happens when the model decides to not respond
                next_prompt = PROMPT_FORCE_MODEL_RESPONSE
                continue

            if last_message["role"] == "assistant" and last_message["content"] and last_message["content"][0]["type"] == "text":
                return last_message["content"][0]["text"]
                # we need to check if the last message is a model response (sometimes model can decide to not respond)

        return "Maximum iterations reached without final response."


    @staticmethod
    async def _fork(process, params, tool_id, last_assistant_response):
        """Fork a conversation"""
        if not process.allow_fork:
            return "Forking is not allowed for this agent, possible reason: You are already a forked instance"

        prompts = params["prompts"]
        print(f"Forking conversation with {len(prompts)} prompts: {prompts}")

        async def process_fork(i, prompt):
            child = process.deepcopy() # TODO: need to implement deepcopy for Process class

            # NOTE: we need to remove all other tool calls from the last assistant response
            # because we might not have the tool call results for other tool calls yet
            # this is also important for the forked process to focus on the assigned goal
            child.messages.append(
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
            child.messages.append(
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
            response = await child.run_till_text_response(
                user_prompt=prompt, max_iterations=20, is_tool_continuation=False
            )
            return {"id": i, "message": response}

        # Process all forks in parallel
        responses = await asyncio.gather(
            *[process_fork(i, prompt) for i, prompt in enumerate(prompts)]
        )
        result = json.dumps(responses)
        return result



