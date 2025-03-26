"""Spawn system call for LLMProcess to create new processes from linked programs."""

import logging
from typing import Any

# Avoid circular import
# LLMProcess is imported within the function

# Set up logger
logger = logging.getLogger(__name__)

# Detailed spawn tool description explaining the Unix metaphor and usage patterns
spawn_tool_description = """
You can use this tool to spawn a specialized process from a linked program to handle specific tasks.
This is analogous to the spawn/exec system calls in Unix where a new process is created to run a different program.

Unlike fork (which creates a copy of the current process), spawn creates a completely new process with:
1. A different system prompt optimized for specific tasks
2. Its own separate conversation history
3. Potentially different tools or capabilities

spawn(program_name, query, additional_preload_files=None)
- program_name: The name of the linked program to call (must be one of the available linked programs)
- query: The query to send to the linked program
- additional_preload_files: Optional list of file paths to preload into the child process's context

The spawn system call will:
1. Create a new process from the specified linked program
2. Preload any specified files into the child process's context (if specified)
3. Send your query to that process
4. Return the process's response to you

When to use this tool:
- When you need specialized expertise that a different system prompt provides
- When you need to delegate a task to a more specialized assistant
- When you need different tools or capabilities than what you currently have
- When you want to keep the current conversation focused on the main task while delegating subtasks
- When you need to share specific file content with the child process

Available programs:
The list of available programs depends on your configuration and will be shown to you when the tool is registered.
"""

# Definition of the spawn tool for Anthropic API
spawn_tool_def = {
    "name": "spawn",
    "description": spawn_tool_description,
    "input_schema": {
        "type": "object",
        "properties": {
            "program_name": {
                "type": "string",
                "description": "Name of the linked program to call",
            },
            "query": {
                "type": "string",
                "description": "The query to send to the linked program",
            },
            "additional_preload_files": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Optional list of file paths to preload into the child process's context",
            },
        },
        "required": ["program_name", "query"],
    },
}


async def spawn_tool(
    program_name: str,
    query: str,
    additional_preload_files: list[str] = None,
    llm_process=None,
) -> dict[str, Any]:
    """Create a new process from a linked program to handle a specific query.

    This system call allows one LLM process to create a new process from a linked program
    to handle specialized tasks, with optional file preloading for context.

    Args:
        program_name: The name of the linked program to call
        query: The query to send to the linked program
        additional_preload_files: Optional list of file paths to preload into the child's context
        llm_process: The parent LLMProcess instance with access to linked programs

    Returns:
        A dictionary with the response from the linked program

    Raises:
        ValueError: If the program_name is not found in linked programs
    """
    if not llm_process or not hasattr(llm_process, "linked_programs"):
        error_msg = "Spawn system call requires a parent LLMProcess with linked_programs defined"
        logger.error(f"SPAWN ERROR: {error_msg}")
        from llmproc.tools.tool_result import ToolResult

        return ToolResult.from_error(error_msg)

    linked_programs = llm_process.linked_programs
    if program_name not in linked_programs:
        available_programs = ", ".join(linked_programs.keys())
        error_msg = f"Program '{program_name}' not found. Available programs: {available_programs}"
        logger.error(f"SPAWN ERROR: {error_msg}")
        from llmproc.tools.tool_result import ToolResult

        return ToolResult.from_error(error_msg)

    try:
        # Get the linked program object
        linked_program = linked_programs[program_name]

        # Check if linked_program is already an LLMProcess or needs instantiation
        if hasattr(linked_program, "run"):
            # It's already a process instance, use it directly
            linked_process = linked_program
        else:
            # It's a Program object, instantiate it as a process
            from llmproc.llm_process import LLMProcess

            linked_process = LLMProcess(program=linked_program)

        # Preload additional files if provided
        if additional_preload_files:
            logger.debug(f"Preloading files for child process: {additional_preload_files}")
            linked_process.preload_files(additional_preload_files)

        # Execute the query on the process
        await linked_process.run(query)

        # Get the actual text response from the process
        response_text = linked_process.get_last_message()

        # Return a successful ToolResult with the response text as content
        from llmproc.tools.tool_result import ToolResult

        return ToolResult.from_success(response_text)
    except Exception as e:

        error_msg = f"Error creating process from program '{program_name}': {str(e)}"
        logger.error(f"SPAWN ERROR: {error_msg}")
        logger.debug("Detailed traceback:", exc_info=True)
        from llmproc.tools.tool_result import ToolResult

        return ToolResult.from_error(error_msg)
