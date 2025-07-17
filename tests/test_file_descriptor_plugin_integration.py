"""Tests for FileDescriptorPlugin integration.

Tests that the FileDescriptorPlugin correctly replicates the embedded FD logic.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from llmproc.common.results import ToolResult
from llmproc.plugins.file_descriptor import FileDescriptorPlugin
from llmproc.plugins.file_descriptor import FileDescriptorManager
from llmproc.program import LLMProgram


class TestFileDescriptorPluginIntegration:
    """Test FileDescriptorPlugin functionality."""

    def test_file_descriptor_plugin_creates_manager(self):
        """Plugin initializes its file descriptor manager."""
        from llmproc.config.schema import FileDescriptorPluginConfig

        plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())

        assert isinstance(plugin.fd_manager, FileDescriptorManager)

    @pytest.mark.asyncio
    async def test_hook_user_input_conversion(self):
        """Large user input is converted via the FD manager."""
        mock_fd_manager = Mock()
        mock_fd_manager.max_input_chars = 5
        mock_fd_manager.handle_user_input.return_value = "fd:1"
        from llmproc.config.schema import FileDescriptorPluginConfig

        plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
        plugin.fd_manager = mock_fd_manager

        process = Mock()
        result = await plugin.hook_user_input("abcdef", process)

        mock_fd_manager.handle_user_input.assert_called_once_with("abcdef")
        assert result == "fd:1"

    @pytest.mark.asyncio
    async def test_hook_tool_result_passthrough_when_small(self):
        """Tool result hook returns None when FD not needed."""
        mock_fd_manager = Mock()
        mock_fd_manager.create_fd_from_tool_result = Mock(return_value=(None, False))
        from llmproc.config.schema import FileDescriptorPluginConfig

        plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
        plugin.fd_manager = mock_fd_manager

        process = Mock()
        result = ToolResult("Test result")
        modified_result = await plugin.hook_tool_result("test_tool", result, process)

        mock_fd_manager.create_fd_from_tool_result.assert_called_once_with("Test result", "test_tool")
        assert modified_result is None

    @pytest.mark.asyncio
    async def test_hook_tool_result_creates_fd_when_enabled(self):
        """Test that tool result hook creates FD when enabled and result is large."""
        # Create mock FD manager that simulates FD creation
        mock_fd_manager = Mock()
        processed_result = ToolResult("FD reference result")
        mock_fd_manager.create_fd_from_tool_result = Mock(return_value=(processed_result, True))
        mock_fd_manager.max_direct_output_chars = 100

        from llmproc.config.schema import FileDescriptorPluginConfig

        plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
        plugin.fd_manager = mock_fd_manager

        process = Mock()
        original_result = ToolResult("Large test result")

        modified_result = await plugin.hook_tool_result("test_tool", original_result, process)

        # Verify FD manager was called with correct parameters
        mock_fd_manager.create_fd_from_tool_result.assert_called_once_with("Large test result", "test_tool")

        # Verify the processed result was returned
        assert modified_result == processed_result

    @pytest.mark.asyncio
    async def test_hook_tool_result_no_fd_when_not_needed(self):
        """Test that tool result hook doesn't create FD when not needed."""
        # Create mock FD manager that returns no FD needed
        mock_fd_manager = Mock()
        mock_fd_manager.create_fd_from_tool_result = Mock(return_value=(None, False))

        from llmproc.config.schema import FileDescriptorPluginConfig

        plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
        plugin.fd_manager = mock_fd_manager

        process = Mock()
        original_result = ToolResult("Small result")

        modified_result = await plugin.hook_tool_result("test_tool", original_result, process)

        # Verify FD manager was called
        mock_fd_manager.create_fd_from_tool_result.assert_called_once_with("Small result", "test_tool")

        # Verify no modification was made
        assert modified_result is None

    @pytest.mark.asyncio
    async def test_hook_tool_result_skips_error_results(self):
        """Test that tool result hook skips error results."""
        mock_fd_manager = Mock()
        from llmproc.config.schema import FileDescriptorPluginConfig

        plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
        plugin.fd_manager = mock_fd_manager

        process = Mock()
        error_result = ToolResult.from_error("Error message")

        modified_result = await plugin.hook_tool_result("test_tool", error_result, process)

        # Verify FD manager was not called for error results
        mock_fd_manager.create_fd_from_tool_result.assert_not_called()

        # Verify no modification was made
        assert modified_result is None

    @pytest.mark.asyncio
    async def test_hook_tool_result_skips_results_without_content(self):
        """Test that tool result hook skips results without content attribute."""
        mock_fd_manager = Mock()
        from llmproc.config.schema import FileDescriptorPluginConfig

        plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
        plugin.fd_manager = mock_fd_manager

        process = Mock()
        # Create a result object with None content
        result_without_content = ToolResult(None)

        modified_result = await plugin.hook_tool_result("test_tool", result_without_content, process)

        # Verify FD manager was not called
        mock_fd_manager.create_fd_from_tool_result.assert_not_called()

        # Verify no modification was made
        assert modified_result is None

    @pytest.mark.asyncio
    async def test_provides_fd_tools(self):
        """FileDescriptorPlugin exposes FD tools through hook_provide_tools."""
        mock_fd_manager = Mock()
        from llmproc.config.schema import FileDescriptorPluginConfig

        plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
        plugin.fd_manager = mock_fd_manager

        program = LLMProgram(model_name="test", provider="test", system_prompt="sp").add_plugins(plugin)
        program.register_tools([])

        config = {
            "fd_manager": mock_fd_manager,
            "linked_programs": {},
            "linked_program_descriptions": {},
            "has_linked_programs": False,
            "provider": "test",
            "mcp_enabled": False,
        }

        from llmproc.tools.tool_manager import ToolManager

        tm = ToolManager()
        await tm.register_tools(program.tools, config)
        assert "read_fd" in tm.registered_tools
        assert "fd_to_file" in tm.registered_tools

    def test_tool_config_override(self):
        """ToolConfig entries override builtin metadata."""
        from llmproc.common.metadata import get_tool_meta, attach_meta
        from llmproc.plugins.override_utils import apply_tool_overrides
        from copy import deepcopy
        from llmproc.config.tool import ToolConfig
        from llmproc.config.schema import FileDescriptorPluginConfig

        plugin = FileDescriptorPlugin(
            FileDescriptorPluginConfig(tools=[ToolConfig(name="read_fd", description="custom")])
        )
        original = deepcopy(get_tool_meta(FileDescriptorPlugin.read_fd_tool))

        tools = apply_tool_overrides(plugin.hook_provide_tools(), plugin.config.tools)
        assert tools == [plugin.read_fd_tool]
        assert get_tool_meta(plugin.read_fd_tool).description == "custom"

        attach_meta(FileDescriptorPlugin.read_fd_tool, original)
