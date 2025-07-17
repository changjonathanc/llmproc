"""Tests for tool call hook integration.

Ensures trigger_tool_call_hook is executed within ToolManager.call_tool."""

import pytest
from unittest.mock import Mock, AsyncMock, NonCallableMock, patch
import asyncio

from llmproc.common.results import ToolResult
from llmproc.plugin.datatypes import ToolCallHookResult
from llmproc.plugin.plugin_event_runner import PluginEventRunner
from llmproc.tools.tool_manager import ToolManager


class TestToolCallHookIntegration:
    """Test tool call hook integration in ToolManager."""

    @pytest.mark.asyncio
    async def test_tool_call_hook_called_from_call_tool(self):
        """Hook should be triggered before tool execution."""
        tool_manager = ToolManager()
        tool_manager.process_access_level = Mock()
        tool_manager.process_access_level.compare_to = Mock(return_value=-1)

        mock_result = ToolResult("Test result")
        tool_manager.runtime_registry = Mock()
        mock_tool = Mock()
        mock_tool.execute = AsyncMock(return_value=mock_result)
        tool_manager.runtime_registry.get_tool = Mock(return_value=mock_tool)

        mock_process = Mock()
        mock_hook_callback = NonCallableMock(spec=["hook_tool_call"])
        mock_hook_callback.hook_tool_call = AsyncMock(return_value=None)
        mock_process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(mock_process._submit_to_loop, [mock_hook_callback])
        mock_process.plugins = runner
        mock_process.hooks = runner

        tool_manager.runtime_context = {"process": mock_process}

        with patch("llmproc.tools.tool_manager.get_tool_meta") as mock_get_meta:
            meta = Mock()
            meta.access = Mock()
            meta.access.compare_to = Mock(return_value=-1)
            meta.requires_context = False
            mock_get_meta.return_value = meta

            result = await tool_manager.call_tool("test_tool", {"arg": "value"})

            mock_hook_callback.hook_tool_call.assert_called_once_with("test_tool", {"arg": "value"}, mock_process)
            mock_tool.execute.assert_awaited_once_with(
                {"arg": "value"},
                runtime_context=tool_manager.runtime_context,
                process_access_level=tool_manager.process_access_level,
            )
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_tool_call_hook_modifies_arguments(self):
        """Hook can modify arguments before execution."""
        tool_manager = ToolManager()
        tool_manager.process_access_level = Mock()
        tool_manager.process_access_level.compare_to = Mock(return_value=-1)

        mock_result = ToolResult("Result")
        tool_manager.runtime_registry = Mock()
        mock_tool = Mock()
        mock_tool.execute = AsyncMock(return_value=mock_result)
        tool_manager.runtime_registry.get_tool = Mock(return_value=mock_tool)

        mock_process = Mock()
        modified_args = {"arg": "modified"}
        hook_result = ToolCallHookResult(modified_args=modified_args)
        mock_hook_callback = NonCallableMock(spec=["hook_tool_call"])
        mock_hook_callback.hook_tool_call = AsyncMock(return_value=hook_result)
        mock_process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(mock_process._submit_to_loop, [mock_hook_callback])
        mock_process.plugins = runner
        mock_process.hooks = runner

        tool_manager.runtime_context = {"process": mock_process}

        with patch("llmproc.tools.tool_manager.get_tool_meta") as mock_get_meta:
            meta = Mock()
            meta.access = Mock()
            meta.access.compare_to = Mock(return_value=-1)
            meta.requires_context = False
            mock_get_meta.return_value = meta

            result = await tool_manager.call_tool("test_tool", {"arg": "value"})

            mock_tool.execute.assert_awaited_once_with(
                modified_args,
                runtime_context=tool_manager.runtime_context,
                process_access_level=tool_manager.process_access_level,
            )
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_tool_call_hook_can_skip_execution(self):
        """Hook can skip tool execution entirely."""
        tool_manager = ToolManager()
        tool_manager.process_access_level = Mock()
        tool_manager.process_access_level.compare_to = Mock(return_value=-1)

        skip_result = ToolResult.from_error("blocked")
        tool_manager.runtime_registry = Mock()
        mock_tool = Mock()
        mock_tool.execute = AsyncMock()
        tool_manager.runtime_registry.get_tool = Mock(return_value=mock_tool)

        mock_process = Mock()
        hook_result = ToolCallHookResult(skip_execution=True, skip_result=skip_result)
        mock_hook_callback = NonCallableMock(spec=["hook_tool_call"])
        mock_hook_callback.hook_tool_call = AsyncMock(return_value=hook_result)
        mock_process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(mock_process._submit_to_loop, [mock_hook_callback])
        mock_process.plugins = runner
        mock_process.hooks = runner

        tool_manager.runtime_context = {"process": mock_process}

        with patch("llmproc.tools.tool_manager.get_tool_meta") as mock_get_meta:
            meta = Mock()
            meta.access = Mock()
            meta.access.compare_to = Mock(return_value=-1)
            meta.requires_context = False
            mock_get_meta.return_value = meta

            result = await tool_manager.call_tool("test_tool", {"arg": "value"})

            mock_tool.execute.assert_not_called()
            assert result == skip_result
