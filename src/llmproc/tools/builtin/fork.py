"""Fork system call for LLMProcess to create a copy of the current process."""

from typing import Any

from llmproc.common.results import ToolResult

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

# Definition of the fork tool for Anthropic API
fork_tool_def = {
    "name": "fork",
    "description": fork_tool_description,
    "input_schema": {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "description": "List of prompts/instructions for each forked process",
                "items": {
                    "type": "string",
                    "description": "A specific task or query to be handled by a forked process",
                },
            }
        },
        "required": ["prompts"],
    },
}


from llmproc.tools.context_aware import context_aware

@context_aware
async def fork_tool(
    prompts: list[str],
    runtime_context=None,
) -> dict[str, Any]:
    """Placeholder function for the fork system call.

    The actual implementation is handled by the process executor,
    as it requires special handling of the process state.

    Args:
        prompts: List of prompts/instructions for each forked process
        runtime_context: Runtime context containing process and other dependencies

    Returns:
        A dictionary with placeholder response
    """
    # This is just a placeholder - the real implementation is in the process executor
    return ToolResult.from_error("Direct calls to fork_tool are not supported. This should be handled by the process executor.")
