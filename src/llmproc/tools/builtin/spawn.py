"""Spawn system call for LLMProcess to create new processes from linked programs."""

import logging
from typing import Any, Optional

from llmproc.common.results import ToolResult
from llmproc.tools.context_aware import context_aware

# Avoid circular import
# LLMProcess is imported within the function

# Set up logger
logger = logging.getLogger(__name__)

# Tool description
SPAWN_DESCRIPTION = """
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

# Schema definition for spawn tool
SPAWN_TOOL_SCHEMA = {
    "name": "spawn",
    "description": SPAWN_DESCRIPTION,
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
                "items": {"type": "string"},
                "description": "Optional list of file paths to preload into the child process's context",
            },
        },
        "required": ["program_name", "query"],
    },
}

# Set default definition
spawn_tool_def = SPAWN_TOOL_SCHEMA


@context_aware
async def spawn_tool(
    program_name: str,
    query: str,
    additional_preload_files: Optional[list[str]] = None,
    runtime_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a new process from a linked program to handle a specific query.

    This system call allows one LLM process to create a new process from a linked program
    to handle specialized tasks, with optional file preloading for context.

    Args:
        program_name: The name of the linked program to call
        query: The query to send to the linked program
        additional_preload_files: Optional list of file paths to preload
        runtime_context: Runtime context dictionary containing dependencies needed by the tool.
            Required keys: 'process' (LLMProcess instance with linked_programs)

    Returns:
        A dictionary with the response from the linked program

    Raises:
        ValueError: If the program_name is not found in linked programs
    """
    # Log arguments for debugging
    logger.debug(
        f"spawn_tool called with args: program_name={program_name}, query={query}, additional_preload_files={additional_preload_files}"
    )

    # Get process from runtime context
    if not runtime_context or "process" not in runtime_context:
        error_msg = "Spawn system call requires runtime_context with process"
        logger.error(f"SPAWN ERROR: {error_msg}")
        return ToolResult.from_error(error_msg)

    # Get process from runtime context
    llm_process = runtime_context["process"]

    if not hasattr(llm_process, "linked_programs"):
        error_msg = "Spawn system call requires a parent LLMProcess with linked_programs defined"
        logger.error(f"SPAWN ERROR: {error_msg}")
        return ToolResult.from_error(error_msg)

    linked_programs = llm_process.linked_programs
    if program_name not in linked_programs:
        # Create a formatted list of available programs with descriptions
        available_programs_list = []
        for name, program in linked_programs.items():
            description = ""
            # Try to get the description from various possible sources
            if (
                hasattr(llm_process, "linked_program_descriptions")
                and name in llm_process.linked_program_descriptions
            ):
                description = llm_process.linked_program_descriptions[name]
            elif hasattr(program, "description") and program.description:
                description = program.description

            if description:
                available_programs_list.append(f"'{name}': {description}")
            else:
                available_programs_list.append(f"'{name}'")

        available_programs = "\n- " + "\n- ".join(available_programs_list)
        error_msg = f"Program '{program_name}' not found. Available programs: {available_programs}"
        logger.error(f"SPAWN ERROR: {error_msg}")
        return ToolResult.from_error(error_msg)

    try:
        # Get the linked program object
        linked_program = linked_programs[program_name]

        # Import from program_exec to ensure consistent process creation
        from llmproc.program_exec import create_process

        # In the new architecture, linked_program should always be an LLMProgram instance
        # We create a process on-demand each time spawn is called
        linked_process = await create_process(linked_program, additional_preload_files)

        # Process file descriptor system if it's available in the parent process
        if hasattr(llm_process, "fd_manager") and llm_process.file_descriptor_enabled:
            # Enable file descriptor system in the child process
            linked_process.file_descriptor_enabled = True

            # Create a FileDescriptorManager for the child if it doesn't already have one
            if (
                not hasattr(linked_process, "fd_manager")
                or linked_process.fd_manager is None
            ):
                from llmproc.file_descriptors import FileDescriptorManager

                linked_process.fd_manager = FileDescriptorManager(
                    default_page_size=llm_process.fd_manager.default_page_size,
                    max_direct_output_chars=llm_process.fd_manager.max_direct_output_chars,
                    max_input_chars=llm_process.fd_manager.max_input_chars,
                    page_user_input=llm_process.fd_manager.page_user_input,
                )

            # Copy settings from parent to child
            linked_process.references_enabled = getattr(
                llm_process, "references_enabled", False
            )

            # Copy all reference file descriptors (they are automatically shared)
            # This enables the child to access references created in the parent
            if linked_process.references_enabled:
                for fd_id, fd_data in llm_process.fd_manager.file_descriptors.items():
                    if (
                        fd_id.startswith("ref:")
                        and fd_id not in linked_process.fd_manager.file_descriptors
                    ):
                        linked_process.fd_manager.file_descriptors[fd_id] = (
                            fd_data.copy()
                        )
                        logger.debug(f"Copied reference {fd_id} to child process")

        # Execute the query on the process
        await linked_process.run(query)

        # Get the actual text response from the process
        response_text = linked_process.get_last_message()

        # Return a successful ToolResult with the response text as content
        return ToolResult.from_success(response_text)
    except Exception as e:
        error_msg = f"Error creating process from program '{program_name}': {str(e)}"
        logger.error(f"SPAWN ERROR: {error_msg}")
        logger.debug("Detailed traceback:", exc_info=True)
        return ToolResult.from_error(error_msg)
