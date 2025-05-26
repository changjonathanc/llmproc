"""Fork system call for LLMProcess to create a copy of the current process.

This is a first-class provider-agnostic tool executed via ToolManager like any other
tool. It follows the Unix-inspired model where tools are accessed through a consistent
interface, and maintains proper causal ordering of messages and tool results.
"""

import asyncio
import copy
import json
import logging
from typing import Any, Optional, Union

from llmproc.common.access_control import AccessLevel
from llmproc.common.results import ToolResult
from llmproc.tools.function_tools import register_tool

# Set up logger
logger = logging.getLogger(__name__)

# Avoid circular import
# LLMProcess is imported within the function

# Detailed fork tool description explaining the Unix metaphor and usage patterns
fork_tool_description = """
You can use this tool to fork the conversation into multiple instances of yourself and let each instance continue answering and using tools.
This is analogous to the fork() system call in Unix.

pid = fork([PROMPT]) # prompt to yourself to continue the conversation

if pid == 0:
    # child process
    You'll receive PROMPT as the tool_result and continue the conversation
else:
    # parent process
    You'll wait for all children to finish and receive the final message from each instance in the following format
    [
        {
            "id": 0,
            "message": "the final message from the child process"
        },
    ]

Different from Unix, you can fork multiple children at once:
fork([PROMPT0, PROMPT1, PROMPT2, ...])

When to use this tool:
You can fork yourself to do tasks that would otherwise fill up the context length but only the final result matters.
For example, if you need to read a large file to find certain details, or if you need to execute multiple tools step by step but you don't need the intermediate results.

You can fork multiple instances to perform tasks in parallel without performing them in serial which would quickly fill up the context length.
Each forked process has a complete copy of the conversation history up to the fork point, ensuring continuity and context preservation.
"""


def tool_result_stub(tool_id: str) -> dict:
    """Create a stub tool result for the child process.

    Args:
        tool_id: The ID of the fork tool call

    Returns:
        A dictionary with the stub tool result
    """
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": "pid==0, you are a child instance produced from a fork. please continue the conversation with only the assigned goal",
            }
        ],
    }


@register_tool(
    name="fork",
    description=fork_tool_description,
    param_descriptions={
        "prompts": "List of prompts/instructions for each forked process. Each item is a specific task or query to be handled by a forked process."
    },
    required=["prompts"],
    requires_context=True,
    required_context_keys=["process", "msg_prefix", "tool_results_prefix", "tool_id"],
    access=AccessLevel.ADMIN,
)
async def fork_tool(
    prompts: list[str],
    runtime_context: Optional[dict[str, Any]] = None,
) -> ToolResult:
    """Implementation of the fork system call.

    Creates multiple child processes with copies of the conversation state up to
    the fork point, and runs them with the provided prompts.

    This tool is executed as a first-class, provider-agnostic tool through the
    ToolManager. It uses the causal buffer mechanism (msg_prefix and tool_results_prefix)
    to maintain proper ordering of messages and tool results.

    Args:
        prompts: List of prompts/instructions for each forked process
        runtime_context: Runtime context containing process, msg_prefix, tool_results_prefix, and tool_id

    Returns:
        ToolResult with the combined results from all child processes
    """
    # Note: The @register_tool decorator already validates runtime_context and access level,
    # so we don't need explicit validation checks here

    # Extract required context values
    parent = runtime_context["process"]
    tool_id = runtime_context["tool_id"]
    msg_prefix = runtime_context["msg_prefix"]
    tool_results_prefix = runtime_context["tool_results_prefix"]

    # Create causal prefix by combining msg_prefix and tool_results_prefix
    # This maintains the correct order of messages and tool results
    prefix = copy.deepcopy(msg_prefix) + copy.deepcopy(tool_results_prefix)

    if not prefix:
        return ToolResult.from_error("Conversation history prefix is empty – cannot fork")

    # Cap the number of children to a reasonable limit
    if len(prompts) > 10:
        return ToolResult.from_error(f"Too many fork children requested ({len(prompts)}). Maximum is 10.")

    logger.info(f"Forking conversation with {len(prompts)} prompts")

    async def run_child(idx, prompt):
        """Create and run a child process with the given prompt."""
        # Use the fork_process method to create a deep copy with WRITE access level
        child = await parent.fork_process(access_level=AccessLevel.WRITE)

        # Inherit history up to fork point (use deep copy to avoid shared references)
        child.state = copy.deepcopy(prefix)

        # Insert stub tool_result recognizing it's a child
        child.state.append(tool_result_stub(tool_id))

        # Ensure the child inherits iteration limits from the parent
        child.max_iterations = getattr(parent, "max_iterations", 10)

        # Use standard run() method instead of directly accessing executors
        # This maintains proper encapsulation and allows the process to handle
        # the provider-specific details internally
        try:
            response = await child.run(prompt, max_iterations=child.max_iterations)

            # Check if we got a string response or a RunResult object
            if not isinstance(response, str):
                logger.info(f"Child {idx} didn't return a string response, getting last message")

                # Use get_last_message() to get the text of the last assistant message
                text_response = child.get_last_message()

                # If we couldn't get a text response, make a follow-up call
                if not text_response:
                    logger.info(f"Child {idx} making follow-up call for final response")
                    follow_up = await child.run(
                        "Please provide a final text summary of your findings.",
                        max_iterations=1,  # Limit to 1 to prevent more tool usage
                    )

                    # Check if the follow-up response is a string
                    if isinstance(follow_up, str):
                        text_response = follow_up
                    else:
                        # Try get_last_message again after follow-up
                        text_response = child.get_last_message()

                # Use the retrieved text or a fallback message
                response = text_response or "No text response available"

            return {"id": idx, "message": response}
        except Exception as e:
            logger.error(f"Error in child process {idx}: {str(e)}", exc_info=True)
            return {"id": idx, "message": f"Error in child process: {str(e)}", "error": True}

    # Process all forks in parallel
    try:
        results = await asyncio.gather(*(run_child(i, p) for i, p in enumerate(prompts)))
        return ToolResult.from_success(json.dumps(results))
    except Exception as e:
        logger.error(f"Error during fork execution: {str(e)}", exc_info=True)
        return ToolResult.from_error(f"Fork error: {str(e)}")
