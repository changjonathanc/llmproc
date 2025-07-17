"""Tests for example plugins.

Tests the example plugins to ensure they demonstrate correct hook usage.
"""

import pytest
from unittest.mock import Mock, patch
from io import StringIO

from llmproc.extensions.examples import (
    TimestampPlugin,
    ToolApprovalPlugin,
    ToolFilterPlugin,
    FileMapPlugin,
)
from llmproc.plugins.message_id import MessageIDPlugin
from llmproc.plugin.datatypes import ToolCallHookResult
from llmproc.common.results import ToolResult
from llmproc.program import LLMProgram


class TestTimestampPlugin:
    """Test the TimestampPlugin."""

    @pytest.mark.asyncio
    async def test_user_input_gets_timestamp(self):
        """Test that user input gets timestamped."""
        plugin = TimestampPlugin()
        process = Mock()

        result = await plugin.hook_user_input("Hello world", process)

        assert result is not None
        assert "Hello world" in result
        assert "[" in result and "]" in result  # Has timestamp format

    @pytest.mark.asyncio
    async def test_custom_timestamp_format(self):
        """Test custom timestamp format."""
        plugin = TimestampPlugin(timestamp_format="%H:%M")
        process = Mock()

        result = await plugin.hook_user_input("Hello", process)

        assert result is not None
        assert "Hello" in result
        # Should have HH:MM format
        import re

        assert re.search(r"\[\d{2}:\d{2}\]", result)

    @pytest.mark.asyncio
    async def test_tool_result_success_gets_timestamp(self):
        """Test that successful tool results get timestamped."""
        plugin = TimestampPlugin()
        process = Mock()

        original_result = ToolResult.from_success("Tool output")
        result = await plugin.hook_tool_result("test_tool", original_result, process)

        assert result is not None
        assert "Tool output" in result.content
        assert "[" in result.content and "]" in result.content

    @pytest.mark.asyncio
    async def test_tool_result_error_unchanged(self):
        """Test that error results are not timestamped."""
        plugin = TimestampPlugin()
        process = Mock()

        original_result = ToolResult.from_error("Error message")
        result = await plugin.hook_tool_result("test_tool", original_result, process)

        assert result is None  # No modification for errors



class TestToolApprovalPlugin:
    """Test the ToolApprovalPlugin."""

    @pytest.mark.asyncio
    async def test_non_restricted_tool_allowed(self):
        """Test that non-restricted tools are allowed through."""
        plugin = ToolApprovalPlugin(approval_required_tools={"spawn"})
        process = Mock()

        result = await plugin.hook_tool_call("read_file", {"path": "test.txt"}, process)

        assert result is None  # Allow execution

    @pytest.mark.asyncio
    async def test_blocked_tool_rejected(self):
        """Test that blocked tools are rejected."""
        plugin = ToolApprovalPlugin()
        plugin.block_list.add("dangerous_tool")
        process = Mock()

        result = await plugin.hook_tool_call("dangerous_tool", {}, process)

        assert result is not None
        assert result.skip_execution is True
        assert result.skip_result.is_error
        assert "blocked by policy" in result.skip_result.content

    @pytest.mark.asyncio
    async def test_auto_approved_tool_allowed(self):
        """Test that auto-approved tools are allowed without prompting."""
        plugin = ToolApprovalPlugin(approval_required_tools={"spawn"})
        plugin.auto_approve_list.add("spawn")
        process = Mock()

        result = await plugin.hook_tool_call("spawn", {"program": "test"}, process)

        assert result is None  # Allow execution


class TestToolFilterPlugin:
    """Test the ToolFilterPlugin."""

    @pytest.mark.asyncio
    async def test_blocked_pattern_rejected(self):
        """Test that blocked patterns are rejected."""
        plugin = ToolFilterPlugin()
        process = Mock()

        result = await plugin.hook_tool_call("read_file", {"path": "/etc/passwd"}, process)

        assert result is not None
        assert result.skip_execution is True
        assert result.skip_result.is_error
        assert "restricted pattern" in result.skip_result.content

    @pytest.mark.asyncio
    async def test_safe_file_allowed(self):
        """Test that safe files are allowed."""
        plugin = ToolFilterPlugin()
        process = Mock()

        result = await plugin.hook_tool_call("read_file", {"path": "/home/user/safe.txt"}, process)

        # Should modify args to add encoding, not block
        assert result is not None
        assert result.skip_execution is False
        assert result.modified_args is not None
        assert result.modified_args["encoding"] == "utf-8"

    @pytest.mark.asyncio
    async def test_read_file_gets_encoding(self):
        """Test that read_file calls get encoding added."""
        plugin = ToolFilterPlugin()
        process = Mock()

        result = await plugin.hook_tool_call("read_file", {"path": "test.txt"}, process)

        assert result is not None
        assert result.modified_args == {"path": "test.txt", "encoding": "utf-8"}

    @pytest.mark.asyncio
    async def test_read_file_with_existing_encoding_unchanged(self):
        """Test that read_file with existing encoding is unchanged."""
        plugin = ToolFilterPlugin()
        process = Mock()

        result = await plugin.hook_tool_call("read_file", {"path": "test.txt", "encoding": "latin-1"}, process)

        assert result is None  # No modification needed


class TestMessageIDPlugin:
    """Test the MessageIDPlugin."""

    @pytest.mark.asyncio
    async def test_user_input_gets_message_id(self):
        """Plugin prefixes message IDs automatically."""
        plugin = MessageIDPlugin()
        process = Mock()
        process.state = ["m1", "m2"]

        result = await plugin.hook_user_input("Hello", process)

        assert result == "[msg_2] Hello"

    def test_provide_tools_returns_goto(self):
        """MessageIDPlugin can provide the goto tool."""
        from llmproc.config.schema import MessageIDPluginConfig

        plugin = MessageIDPlugin(MessageIDPluginConfig(enable_goto=True))
        provided = plugin.hook_provide_tools()

        assert len(provided) == 1
        assert provided[0].__name__ == 'goto_tool'


class TestFileMapPlugin:
    """Test the FileMapPlugin."""

    @pytest.mark.asyncio
    async def test_file_map_in_system_prompt(self, tmp_path):
        """Plugin adds file map listing."""
        (tmp_path / "a.txt").write_text("content")
        plugin = FileMapPlugin(root=str(tmp_path), max_files=1, show_size=False)
        process = Mock()

        result = await plugin.hook_system_prompt("base", process)

        assert "file_map:" in result
        assert "a.txt" in result
