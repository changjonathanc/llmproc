"""Example plugins demonstrating the hook system.

These plugins show how to use the hook system for behavior modification
and serve as examples for users creating their own plugins.
"""

import datetime
import logging
import os
import subprocess
from pathlib import Path

from llmproc.common.results import ToolResult
from llmproc.plugin.datatypes import ToolCallHookResult

logger = logging.getLogger(__name__)


class TimestampPlugin:
    """Plugin that adds timestamps to user input and tool results."""

    def __init__(self, timestamp_format: str = "%Y-%m-%d %H:%M:%S"):
        """Initialize the timestamp plugin.

        Args:
            timestamp_format: strftime format for timestamps
        """
        self.timestamp_format = timestamp_format

    def fork(self) -> "TimestampPlugin":
        """Return ``self`` for a forked process."""
        return self

    async def hook_user_input(self, user_input: str, process) -> str | None:
        """Add timestamp to user input."""
        timestamp = datetime.datetime.now().strftime(self.timestamp_format)
        return f"[{timestamp}] {user_input}"

    async def hook_tool_result(self, tool_name: str, result: ToolResult, process) -> ToolResult | None:
        """Add timestamp to successful tool results."""
        if not result.is_error:
            timestamp = datetime.datetime.now().strftime(self.timestamp_format)
            modified_content = f"[{timestamp}] {result.content}"
            return ToolResult(modified_content)
        return None


class ToolApprovalPlugin:
    """Plugin that requires user approval for certain tools."""

    def __init__(self, approval_required_tools: set[str] = None):
        """Initialize the tool approval plugin.

        Args:
            approval_required_tools: Set of tool names that require approval
        """
        self.approval_required_tools = approval_required_tools or {"spawn", "fd_to_file"}
        self.auto_approve_list = set()  # Tools that are pre-approved for this session
        self.block_list = set()  # Tools that are blocked

    def fork(self) -> "ToolApprovalPlugin":
        """Return ``self`` for a forked process."""
        return self

    async def hook_tool_call(self, tool_name: str, args: dict, process) -> ToolCallHookResult | None:
        """Gate tool calls through approval system."""
        # Check block list
        if tool_name in self.block_list:
            return ToolCallHookResult(
                skip_execution=True, skip_result=ToolResult.from_error(f"Tool '{tool_name}' is blocked by policy")
            )

        # Check if approval required
        if tool_name in self.approval_required_tools and tool_name not in self.auto_approve_list:
            # Prompt user for approval
            approval = await self._request_user_approval(tool_name, args)

            if not approval:
                return ToolCallHookResult(
                    skip_execution=True, skip_result=ToolResult.from_error(f"Tool '{tool_name}' rejected by user")
                )

            # Add to auto-approve for this session
            self.auto_approve_list.add(tool_name)

        return None  # Allow execution

    async def _request_user_approval(self, tool_name: str, args: dict) -> bool:
        """Request user approval for tool execution."""
        print("\n🤖 Tool Call Approval Required:")
        print(f"   Tool: {tool_name}")
        print(f"   Args: {args}")

        while True:
            response = input("Approve this tool call? (y/n/always/block): ").strip().lower()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            elif response in ["always", "a"]:
                self.auto_approve_list.add(tool_name)
                return True
            elif response in ["block", "b"]:
                self.block_list.add(tool_name)
                return False
            print("Please enter 'y', 'n', 'always', or 'block'")


class ToolFilterPlugin:
    """Plugin that filters tool arguments or blocks specific patterns."""

    def __init__(self):
        """Initialize the tool filter plugin."""
        self.blocked_patterns = {
            "read_file": ["/etc/passwd", "/etc/shadow"],  # Block sensitive files
            "spawn": ["rm -rf", "sudo"],  # Block dangerous commands
        }

    def fork(self) -> "ToolFilterPlugin":
        """Return ``self`` for a forked process."""
        return self

    async def hook_tool_call(self, tool_name: str, args: dict, process) -> ToolCallHookResult | None:
        """Filter tool calls based on patterns."""
        if tool_name in self.blocked_patterns:
            patterns = self.blocked_patterns[tool_name]

            # Check if any blocked pattern is in the arguments
            args_str = str(args).lower()
            for pattern in patterns:
                if pattern in args_str:
                    return ToolCallHookResult(
                        skip_execution=True,
                        skip_result=ToolResult.from_error(
                            f"Tool call blocked: contains restricted pattern '{pattern}'"
                        ),
                    )

        # Example: Auto-add encoding to read_file calls
        if tool_name == "read_file" and "encoding" not in args:
            modified_args = {**args, "encoding": "utf-8"}
            return ToolCallHookResult(modified_args=modified_args)

        return None  # Allow execution


class EnvVarInfoPlugin:
    """Example plugin that appends environment variables to the system prompt."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def fork(self) -> "EnvVarInfoPlugin":
        """Return ``self`` for a forked process."""
        return self

    async def hook_system_prompt(self, system_prompt: str, process) -> str | None:
        lines = []
        for label, var_name in self.mapping.items():
            value = os.getenv(var_name)
            if value:
                lines.append(f"{label}: {value.rstrip()}")
        if not lines:
            return None
        env_block = "<env>\n" + "\n".join(lines) + "\n</env>"
        return f"{system_prompt}\n\n{env_block}"


class CommandInfoPlugin:
    """Example plugin that appends command output to the system prompt."""

    def __init__(self, commands: list[str]):
        self.commands = commands

    def fork(self) -> "CommandInfoPlugin":
        """Return ``self`` for a forked process."""
        return self

    async def hook_system_prompt(self, system_prompt: str, process) -> str | None:
        lines = []
        for i, cmd in enumerate(self.commands):
            lines.append(f"> {cmd}")
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                output = result.stdout.strip()
                if output:
                    lines.extend(output.splitlines())
                if result.returncode != 0:
                    lines.append(f"error({result.returncode})")
            except Exception:
                lines.append("error")
            if i < len(self.commands) - 1:
                lines.append("")
        if not lines:
            return None
        env_block = "<env>\n" + "\n".join(lines) + "\n</env>"
        return f"{system_prompt}\n\n{env_block}"

    async def hook_tool_result(self, tool_name: str, result: ToolResult, process) -> ToolResult | None:
        """Convert large tool outputs to file descriptors."""
        if not self.enabled:
            return None

        if not result.is_error and hasattr(result, "content") and result.content is not None:
            # Use the exact same logic as the embedded FD system
            processed_result, used_fd = self.fd_manager.create_fd_from_tool_result(result.content, tool_name)
            if used_fd:
                logger.info(
                    f"Tool result from '{tool_name}' exceeds {self.fd_manager.max_direct_output_chars} chars, creating file descriptor"
                )
                logger.debug(f"Created file descriptor for tool result from '{tool_name}'")
                return processed_result
        return None


class FileMapPlugin:
    """Example plugin that lists files in a directory."""

    def __init__(self, root: str = ".", max_files: int = 50, show_size: bool = True):
        self.root = Path(root)
        self.max_files = max_files
        self.show_size = show_size

    def fork(self) -> "FileMapPlugin":
        """Return ``self`` for a forked process."""
        return self

    async def hook_system_prompt(self, system_prompt: str, process) -> str | None:
        root_dir = self.root.resolve()
        if not root_dir.exists() or not root_dir.is_dir():
            logger.warning("FileMapPlugin root directory not found: %s", root_dir)
            return None

        files = [p for p in root_dir.rglob("*") if p.is_file()]
        lines = ["file_map:"]
        for path in files[: self.max_files]:
            rel = path.relative_to(root_dir)
            size_part = f" ({path.stat().st_size} bytes)" if self.show_size else ""
            lines.append(f"  {rel}{size_part}")
        if len(files) > self.max_files:
            lines.append(f"  ... ({len(files) - self.max_files} more files)")

        env_block = "<env>\n" + "\n".join(lines) + "\n</env>"
        return f"{system_prompt}\n\n{env_block}"
