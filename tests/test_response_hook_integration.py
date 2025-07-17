"""Tests for response hook integration in LLMProcess."""

from unittest.mock import AsyncMock, Mock, patch
import asyncio

import pytest

from llmproc.plugin.plugin_event_runner import PluginEventRunner


class TestResponseHookIntegration:
    """Test response hook integration in LLMProcess."""

    @pytest.mark.asyncio
    async def test_hook_invoked_by_executor(self):
        """Executor should trigger response hooks during streaming."""
        process = Mock()
        process.trigger_event = AsyncMock()
        process._process_user_input = Mock(return_value="processed")

        async def fake_run(proc, user_input, max_iter):
            await proc.hooks.response(proc, "chunk")
            return Mock()

        process.executor = Mock()
        process.executor.run = AsyncMock(side_effect=fake_run)
        process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(process._submit_to_loop, [])
        process.plugins = runner
        process.hooks = runner

        with patch.object(process.hooks, "response", AsyncMock(return_value=None)) as mock_hook:
            from llmproc.llm_process import LLMProcess

            await LLMProcess._async_run(process, "prompt", 5)

            mock_hook.assert_awaited_once_with(process, "chunk")
