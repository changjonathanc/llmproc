from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from llmproc.common.access_control import AccessLevel
from llmproc.common.results import ToolResult
from llmproc.tools.function_tools import register_tool

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from llmproc.llm_process import LLMProcess

logger = logging.getLogger(__name__)

SPAWN_DESCRIPTION = """You can use this tool to spawn a specialized process from a linked program to handle specific tasks.
This is analogous to the spawn/exec system calls in Unix where a new process is created to run a different program.

Unlike fork (which creates a copy of the current process), spawn creates a completely new process with:
1. A different system prompt optimized for specific tasks
2. Its own separate conversation history
3. Potentially different tools or capabilities

- spawn(program_name, prompt, additional_preload_files=None)
- program_name: The name of the linked program to call. Leave blank to spawn the current program when no linked programs are configured.
- prompt: The prompt to send to the linked program
- additional_preload_files: Optional list of file paths to preload into the child process's context

The spawn system call will:
1. Create a new process from the specified linked program
2. Preload any specified files into the child process's context (if specified)
3. Send your prompt to that process
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


def modify_spawn_schema(schema: dict, config: dict) -> dict:
    """Modify spawn tool schema with linked program details."""
    linked_programs = config.get("linked_programs", {})
    linked_program_descriptions = config.get("linked_program_descriptions", {})

    if linked_programs:
        available_programs_list = []
        if linked_program_descriptions:
            for name, description in linked_program_descriptions.items():
                if name in linked_programs:
                    available_programs_list.append(f"'{name}': {description}")
        for name in linked_programs:
            if not (linked_program_descriptions and name in linked_program_descriptions):
                available_programs_list.append(f"'{name}'")
        if available_programs_list:
            formatted_programs = "\n\n## Available Programs:\n- " + "\n- ".join(available_programs_list)
            schema["description"] += formatted_programs

    return schema


class SpawnPlugin:
    """Plugin that registers the spawn tool and stores linked programs."""

    def __init__(
        self,
        linked_programs: Optional[dict[str, Any]] = None,
        linked_program_descriptions: Optional[dict[str, str]] = None,
    ) -> None:
        self.linked_programs = linked_programs or {}
        self.linked_program_descriptions = linked_program_descriptions or {}

    def fork(self) -> SpawnPlugin:
        return SpawnPlugin(
            linked_programs=self.linked_programs.copy(),
            linked_program_descriptions=self.linked_program_descriptions.copy(),
        )

    def hook_provide_tools(self) -> list:
        return [self.spawn_tool]

    def _format_available_programs(self) -> str:
        """Create a formatted string listing available linked programs."""
        available_programs_list: list[str] = []
        for name, program in self.linked_programs.items():
            description = ""
            if name in self.linked_program_descriptions:
                description = self.linked_program_descriptions[name]
            elif hasattr(program, "description") and program.description:
                description = program.description
            available_programs_list.append(f"'{name}': {description}" if description else f"'{name}'")
        return "\n- " + "\n- ".join(available_programs_list)

    def _validate_spawn_inputs(self, llm_process: Any, program_name: str) -> tuple[bool, Optional[str]]:
        """Validate inputs and decide whether to spawn the current program."""
        spawn_self = not program_name or not self.linked_programs
        if not spawn_self and program_name not in self.linked_programs:
            available_programs = self._format_available_programs()
            error_msg = f"Program '{program_name}' not found. Available programs: {available_programs}"
            return False, error_msg

        return spawn_self, None

    async def spawn_tool(
        self,
        prompt: str,
        program_name: str = "",
        additional_preload_files: Optional[list[str]] = None,
        runtime_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a new process from a linked program to handle a specific prompt."""
        logger.debug(
            "spawn_tool called with args: program_name=%s, prompt=%s, additional_preload_files=%s",
            program_name,
            prompt,
            additional_preload_files,
        )

        llm_process = runtime_context["process"]

        spawn_self, validation_error = self._validate_spawn_inputs(llm_process, program_name)
        if validation_error:
            logger.error("Tool 'spawn' error: %s", validation_error)
            return ToolResult.from_error(validation_error)

        try:
            linked_program = llm_process.program if spawn_self else self.linked_programs[program_name]
            from llmproc.program_exec import create_process

            linked_process = await create_process(linked_program)

            if additional_preload_files:
                from llmproc.plugins.preload_files import build_preload_content, load_files

                parent_process = runtime_context.get("process")
                base_dir = getattr(parent_process, "base_dir", None) or Path.cwd()
                content = load_files(additional_preload_files, base_dir)
                if content:
                    preload_section = build_preload_content(content)
                    linked_process.enriched_system_prompt += f"\n\n{preload_section}"
                    logger.debug("Added %d preload files to spawned process", len(additional_preload_files))

            linked_process.access_level = AccessLevel.WRITE
            if hasattr(linked_process, "tool_manager") and linked_process.tool_manager:
                linked_process.tool_manager.set_process_access_level(AccessLevel.WRITE)

            await linked_process.run(prompt)
            response_text = linked_process.get_last_message()
            return ToolResult.from_success(response_text)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error creating process from program '{program_name}': {str(e)}"
            logger.error("SPAWN ERROR: %s", error_msg)
            logger.debug("Detailed traceback:", exc_info=True)
            return ToolResult.from_error(error_msg)


@register_tool(
    name="spawn",
    description=SPAWN_DESCRIPTION,
    param_descriptions={
        "program_name": "Name of the linked program to call. Leave blank to spawn the current program",
        "prompt": "The prompt to send to the linked program",
        "additional_preload_files": "Optional list of file paths to preload into the child process's context",
    },
    required=["prompt"],
    requires_context=True,
    schema_modifier=modify_spawn_schema,
    access=AccessLevel.ADMIN,
)
async def spawn_tool(
    prompt: str,
    program_name: str = "",
    additional_preload_files: Optional[list[str]] = None,
    runtime_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Wrapper that forwards to the :class:`SpawnPlugin` instance."""
    if not runtime_context or "process" not in runtime_context:
        return ToolResult.from_error("spawn requires runtime context")

    process: LLMProcess = runtime_context["process"]
    plugin: SpawnPlugin | None = None
    if hasattr(process, "get_plugin"):
        plugin = process.get_plugin(SpawnPlugin)

    if not plugin:
        plugin = SpawnPlugin()

    return await plugin.spawn_tool(
        prompt,
        program_name=program_name,
        additional_preload_files=additional_preload_files,
        runtime_context=runtime_context,
    )


__all__ = ["SpawnPlugin", "spawn_tool", "modify_spawn_schema", "SPAWN_DESCRIPTION"]
