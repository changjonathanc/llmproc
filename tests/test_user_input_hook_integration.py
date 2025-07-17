"""Tests for user input hook integration in LLMProcess.

Tests that user input hooks are properly called from LLMProcess._async_run().
"""

from unittest.mock import AsyncMock, Mock, patch
import asyncio

import pytest

from llmproc.plugin.plugin_event_runner import PluginEventRunner


class TestUserInputHookIntegration:
    """Test user input hook integration in LLMProcess."""

    @pytest.mark.asyncio
    async def test_trigger_user_input_hook_called_from_async_run(self):
        """Test that trigger_user_input_hook is called from _async_run."""
        # Create a mock LLMProcess with minimal required attributes
        process = Mock()
        process.trigger_event = AsyncMock()
        process._process_user_input = Mock(return_value="processed_input")
        process.executor = Mock()
        process.executor.run = AsyncMock(return_value=Mock())
        process.state = []

        process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(process._submit_to_loop, [])
        process.plugins = runner
        process.hooks = runner
        with patch.object(process.hooks, "user_input", AsyncMock(return_value="hooked_input")) as mock_hook:
            from llmproc.llm_process import LLMProcess

            await LLMProcess._async_run(process, "original_input", 10)

            mock_hook.assert_awaited_once_with("original_input", process)

            # Verify _process_user_input was called with hooked input
            process._process_user_input.assert_called_once_with("hooked_input")

    @pytest.mark.asyncio
    async def test_user_input_flows_through_hook_system(self):
        """Test that user input flows correctly through the hook system."""
        # Create a mock plugin that modifies input
        plugin = Mock()

        async def mock_hook(user_input, process):
            return f"[MODIFIED] {user_input}"

        plugin.hook_user_input = mock_hook

        # Test the hook function directly first
        process = Mock()
        process.trigger_event = AsyncMock()
        process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(process._submit_to_loop, [plugin])
        process.plugins = runner
        process.hooks = runner

        result = await process.hooks.user_input("test input", process)
        assert result == "[MODIFIED] test input"

    @pytest.mark.asyncio
    async def test_no_hooks_returns_original_input(self):
        """Test that when no hooks are present, original input is returned."""
        process = Mock()
        process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(process._submit_to_loop, [])
        process.plugins = runner
        process.hooks = runner

        result = await process.hooks.user_input("original input", process)
        assert result == "original input"

    @pytest.mark.asyncio
    async def test_multiple_hooks_chained(self):
        """Test that multiple user input hooks are chained correctly."""
        # First hook adds timestamp
        plugin1 = Mock()

        async def timestamp_hook(user_input, process):
            return f"[12:00] {user_input}"

        plugin1.hook_user_input = timestamp_hook

        # Second hook adds prefix
        plugin2 = Mock()

        async def prefix_hook(user_input, process):
            return f"PREFIX: {user_input}"

        plugin2.hook_user_input = prefix_hook

        process = Mock()
        process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(process._submit_to_loop, [plugin1, plugin2])
        process.plugins = runner
        process.hooks = runner

        result = await process.hooks.user_input("hello", process)
        assert result == "PREFIX: [12:00] hello"
