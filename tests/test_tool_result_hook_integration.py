"""Tests for tool result hook integration.

Tests that tool result hooks are properly integrated into ToolManager.call_tool().
"""

import pytest
from unittest.mock import Mock, AsyncMock, NonCallableMock
import asyncio

from llmproc.common.results import ToolResult
from llmproc.tools.tool_manager import ToolManager
from llmproc.plugin.plugin_event_runner import PluginEventRunner


class TestToolResultHookIntegration:
    """Test tool result hook integration in ToolManager."""

    @pytest.mark.asyncio
    async def test_tool_result_hook_called_from_call_tool(self):
        """Test that trigger_tool_result_hook is called from call_tool."""
        # Create a mock tool manager with minimal setup
        tool_manager = ToolManager()
        tool_manager.process_access_level = Mock()
        tool_manager.process_access_level.compare_to = Mock(return_value=-1)  # Allow access

        # Create a mock tool result
        mock_result = ToolResult("Test result")

        # Mock the registry to return our test result
        tool_manager.runtime_registry = Mock()
        mock_tool = Mock()
        mock_tool.execute = AsyncMock(return_value=mock_result)
        tool_manager.runtime_registry.get_tool = Mock(return_value=mock_tool)

        # Mock the process with plugins
        mock_process = Mock()
        mock_hook_callback = NonCallableMock(spec=["hook_tool_result"])
        mock_hook_callback.hook_tool_result = AsyncMock(return_value=None)
        mock_process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(mock_process._submit_to_loop, [mock_hook_callback])
        mock_process.plugins = runner
        mock_process.hooks = runner

        # Set up runtime context
        tool_manager.runtime_context = {"process": mock_process}

        # Mock tool metadata
        from llmproc.common.metadata import get_tool_meta
        from unittest.mock import patch

        with patch("llmproc.tools.tool_manager.get_tool_meta") as mock_get_meta:
            meta = Mock()
            meta.access = Mock()
            meta.access.compare_to = Mock(return_value=-1)
            meta.requires_context = False
            mock_get_meta.return_value = meta

            # Call the tool
            result = await tool_manager.call_tool("test_tool", {"arg": "value"})

            # Verify the hook was called
            mock_hook_callback.hook_tool_result.assert_called_once_with("test_tool", mock_result, mock_process)

            # Verify the result was returned
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_tool_result_hook_modifies_result(self):
        """Test that tool result hooks can modify the result."""
        # Create a mock tool manager with minimal setup
        tool_manager = ToolManager()
        tool_manager.process_access_level = Mock()
        tool_manager.process_access_level.compare_to = Mock(return_value=-1)

        # Create original and modified results
        original_result = ToolResult("Original result")
        modified_result = ToolResult("Modified result")

        # Mock the registry
        tool_manager.runtime_registry = Mock()
        mock_tool = Mock()
        mock_tool.execute = AsyncMock(return_value=original_result)
        tool_manager.runtime_registry.get_tool = Mock(return_value=mock_tool)

        # Mock the process with hook that modifies result
        mock_process = Mock()
        mock_hook_callback = NonCallableMock(spec=["hook_tool_result"])
        mock_hook_callback.hook_tool_result = AsyncMock(return_value=modified_result)
        mock_process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(mock_process._submit_to_loop, [mock_hook_callback])
        mock_process.plugins = runner
        mock_process.hooks = runner

        # Set up runtime context
        tool_manager.runtime_context = {"process": mock_process}

        # Mock tool metadata
        from llmproc.common.metadata import get_tool_meta
        from unittest.mock import patch

        with patch("llmproc.tools.tool_manager.get_tool_meta") as mock_get_meta:
            meta = Mock()
            meta.access = Mock()
            meta.access.compare_to = Mock(return_value=-1)
            meta.requires_context = False
            mock_get_meta.return_value = meta

            # Call the tool
            result = await tool_manager.call_tool("test_tool", {"arg": "value"})

            # Verify the hook was called with original result
            mock_hook_callback.hook_tool_result.assert_called_once_with("test_tool", original_result, mock_process)

            # Verify the modified result was returned
            assert result == modified_result

    @pytest.mark.asyncio
    async def test_tool_result_hook_skipped_when_no_process(self):
        """Test that tool result hooks are skipped when no process is available."""
        # Create a mock tool manager with minimal setup
        tool_manager = ToolManager()
        tool_manager.process_access_level = Mock()
        tool_manager.process_access_level.compare_to = Mock(return_value=-1)

        # Create a mock tool result
        mock_result = ToolResult("Test result")

        # Mock the registry
        tool_manager.runtime_registry = Mock()
        mock_tool = Mock()
        mock_tool.execute = AsyncMock(return_value=mock_result)
        tool_manager.runtime_registry.get_tool = Mock(return_value=mock_tool)

        # Set up runtime context WITHOUT process
        tool_manager.runtime_context = {}

        # Mock tool metadata
        from llmproc.common.metadata import get_tool_meta
        from unittest.mock import patch

        with patch("llmproc.tools.tool_manager.get_tool_meta") as mock_get_meta:
            meta = Mock()
            meta.access = Mock()
            meta.access.compare_to = Mock(return_value=-1)
            meta.requires_context = False
            mock_get_meta.return_value = meta

            # Call the tool
            result = await tool_manager.call_tool("test_tool", {"arg": "value"})

            # Verify the original result was returned (no hooks called)
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_fd_logic_now_handled_by_hooks(self):
        """Test that FD logic is now handled by hooks instead of embedded logic."""
        # Create a mock tool manager with minimal setup
        tool_manager = ToolManager()
        tool_manager.process_access_level = Mock()
        tool_manager.process_access_level.compare_to = Mock(return_value=-1)

        # Create a mock tool result
        original_result = ToolResult("Test result")
        fd_processed_result = ToolResult("FD processed result")

        # Mock the registry
        tool_manager.runtime_registry = Mock()
        mock_tool = Mock()
        mock_tool.execute = AsyncMock(return_value=original_result)
        tool_manager.runtime_registry.get_tool = Mock(return_value=mock_tool)

        # Mock process with FD plugin that processes results
        mock_process = Mock()
        mock_fd_plugin = NonCallableMock(spec=["hook_tool_result"])
        mock_fd_plugin.hook_tool_result = AsyncMock(return_value=fd_processed_result)
        mock_process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(mock_process._submit_to_loop, [mock_fd_plugin])
        mock_process.plugins = runner
        mock_process.hooks = runner

        # Set up runtime context with process (hooks enabled)
        tool_manager.runtime_context = {"process": mock_process}

        # Mock tool metadata
        from llmproc.common.metadata import get_tool_meta
        from unittest.mock import patch

        with patch("llmproc.tools.tool_manager.get_tool_meta") as mock_get_meta:
            meta = Mock()
            meta.access = Mock()
            meta.access.compare_to = Mock(return_value=-1)
            meta.requires_context = False
            mock_get_meta.return_value = meta

            # Call the tool
            result = await tool_manager.call_tool("test_tool", {"arg": "value"})

            # Verify FD hook was called
            mock_fd_plugin.hook_tool_result.assert_called_once_with("test_tool", original_result, mock_process)

            # Verify the FD processed result was returned
            assert result == fd_processed_result
