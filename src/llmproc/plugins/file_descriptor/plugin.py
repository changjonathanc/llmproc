from __future__ import annotations

import logging

from llmproc.common.results import ToolResult
from llmproc.config.schema import FileDescriptorPluginConfig
from llmproc.tools.function_tools import register_tool

from .constants import (
    FILE_DESCRIPTOR_INSTRUCTIONS as file_descriptor_instructions,
)
from .constants import (
    REFERENCE_INSTRUCTIONS as reference_instructions,
)
from .constants import (
    USER_INPUT_INSTRUCTIONS as fd_user_input_instructions,
)
from .manager import FileDescriptorManager

logger = logging.getLogger(__name__)


class FileDescriptorPlugin:
    """Plugin that converts large inputs/outputs to file descriptors."""

    def __init__(self, config: FileDescriptorPluginConfig) -> None:
        self.config = config
        self.fd_manager = FileDescriptorManager(
            default_page_size=config.default_page_size,
            max_direct_output_chars=config.max_direct_output_chars,
            max_input_chars=config.max_input_chars,
            page_user_input=config.page_user_input,
            enable_references=config.enable_references,
        )

    def fork(self) -> FileDescriptorPlugin:
        cloned_cfg = FileDescriptorPluginConfig(**self.config.model_dump())
        cloned = FileDescriptorPlugin(cloned_cfg)
        return cloned

    async def hook_user_input(self, user_input: str, process) -> str | None:
        if len(user_input) > self.fd_manager.max_input_chars:
            # The manager now exposes handle_user_input() instead of store().
            # It returns either the original input or a formatted reference to
            # the created file descriptor.
            processed = self.fd_manager.handle_user_input(user_input)
            logger.info("Large user input (%s chars) converted to file descriptor", len(user_input))
            return processed
        return None

    async def hook_system_prompt(self, system_prompt: str, process) -> str | None:
        parts = [system_prompt, file_descriptor_instructions]
        if getattr(self.fd_manager, "page_user_input", False):
            parts.append(fd_user_input_instructions)
        if getattr(self.fd_manager, "enable_references", False):
            parts.append(reference_instructions)
        return "\n\n".join(parts)

    async def hook_tool_result(self, tool_name: str, result: ToolResult, process) -> ToolResult | None:
        if not result.is_error and getattr(result, "content", None) is not None:
            processed_result, used_fd = self.fd_manager.create_fd_from_tool_result(result.content, tool_name)
            if used_fd:
                logger.info(
                    "Tool result from '%s' exceeds %s chars, creating file descriptor",
                    tool_name,
                    self.fd_manager.max_direct_output_chars,
                )
                logger.debug("Created file descriptor for tool result from '%s'", tool_name)
                return processed_result
        return None

    async def hook_response(self, response: str, process) -> str | None:
        if not getattr(self.fd_manager, "enable_references", False):
            return None
        if isinstance(response, list):
            for block in response:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_content = block.get("text", "")
                    self.fd_manager.process_references(text_content)
                elif isinstance(block, str):
                    self.fd_manager.process_references(block)
        else:
            self.fd_manager.process_references(response)
        return None

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------
    @register_tool(
        name="read_fd",
        description="Read content from a file descriptor with paging and extraction options.",
        param_descriptions={
            "fd": "File descriptor ID to read from (e.g., 'fd:12345' or 'ref:example_id')",
            "read_all": "If true, returns the entire content regardless of size",
            "extract_to_new_fd": "If true, extracts content to a new file descriptor instead of returning directly",
            "mode": "Positioning mode: 'page' (default), 'line', or 'char'",
            "start": "Starting position (page number, line number, or character position)",
            "count": "Number of units to read (pages, lines, or characters)",
        },
        required=["fd"],
    )
    async def read_fd_tool(
        self,
        fd: str,
        read_all: bool = False,
        extract_to_new_fd: bool = False,
        mode: str = "page",
        start: int = 1,
        count: int = 1,
    ) -> ToolResult:
        try:
            xml_content = self.fd_manager.read_fd_content(
                fd_id=fd,
                read_all=read_all,
                extract_to_new_fd=extract_to_new_fd,
                mode=mode,
                start=start,
                count=count,
            )
            return ToolResult.from_success(xml_content)
        except KeyError as e:
            from .formatter import format_fd_error

            xml_error = format_fd_error("not_found", fd, str(e))
            return ToolResult.from_error(xml_error)
        except ValueError as e:
            from .formatter import format_fd_error

            xml_error = format_fd_error("invalid_page", fd, str(e))
            return ToolResult.from_error(xml_error)
        except Exception as e:  # pragma: no cover - defensive
            from .formatter import format_fd_error

            logger.error("Tool 'read_fd' error: %s", e)
            logger.debug("Detailed traceback:", exc_info=True)
            xml_error = format_fd_error("read_error", fd, f"Error reading file descriptor: {e}")
            return ToolResult.from_error(xml_error)

    @register_tool(
        name="fd_to_file",
        description="Write file descriptor content to a file on disk.",
        param_descriptions={
            "fd": "File descriptor ID to export (e.g., 'fd:12345' or 'ref:example_id')",
            "file_path": "Absolute path to the file to write",
            "mode": "Write mode: 'write' (default) or 'append'",
            "create": "Create file if it doesn't exist (default: True)",
            "exist_ok": "Allow overwriting existing file (default: True)",
        },
        required=["fd", "file_path"],
    )
    async def fd_to_file_tool(
        self,
        fd: str,
        file_path: str,
        mode: str = "write",
        create: bool = True,
        exist_ok: bool = True,
    ) -> ToolResult:
        try:
            xml_content = self.fd_manager.write_fd_to_file_content(
                fd_id=fd,
                file_path=file_path,
                mode=mode,
                create=create,
                exist_ok=exist_ok,
            )
            return ToolResult.from_success(xml_content)
        except KeyError as e:
            from .formatter import format_fd_error

            xml_error = format_fd_error("not_found", fd, str(e))
            return ToolResult.from_error(xml_error)
        except ValueError as e:
            from .formatter import format_fd_error

            xml_error = format_fd_error("invalid_parameter", fd, str(e))
            return ToolResult.from_error(xml_error)
        except Exception as e:  # pragma: no cover - defensive
            from .formatter import format_fd_error

            logger.error("Tool 'fd_to_file' error: %s", e)
            logger.debug("Detailed traceback:", exc_info=True)
            xml_error = format_fd_error("write_error", fd, f"Error writing file descriptor to file: {e}")
            return ToolResult.from_error(xml_error)

    def hook_provide_tools(self) -> list:
        """Return the FD-related tools."""
        return [self.read_fd_tool, self.fd_to_file_tool]

    # File descriptor inheritance removed for simplicity
    # Child processes start with clean FD state
    # Users can explicitly share data via preload files if needed
