import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmproc.common.results import RunResult, ToolResult
from llmproc.plugin.events import CallbackEvent
from llmproc.plugin.plugin_event_runner import PluginEventRunner
from llmproc.providers.anthropic_process_executor import (
    AnthropicProcessExecutor,
    IterationState,
)


@pytest.mark.asyncio
async def test_api_stream_block_callback_invoked():
    process = MagicMock()
    process.model_name = "claude-3-haiku"
    process.provider = "anthropic"
    process.enriched_system_prompt = "system"
    process.state = []
    process.tool_manager = MagicMock(runtime_context={})
    process.api_params = {}
    process.client = MagicMock()
    process.trigger_event = AsyncMock()
    fd_plugin = MagicMock()
    fd_plugin.fd_manager = MagicMock(max_direct_output_chars=8000)
    process.get_plugin = MagicMock(return_value=fd_plugin)
    process.call_tool = AsyncMock(return_value=ToolResult.from_success("ok"))

    tasks = []
    process._submit_to_loop = lambda coro: tasks.append(asyncio.get_running_loop().create_task(coro))
    runner = PluginEventRunner(process._submit_to_loop, [])
    process.plugins = runner
    process.hooks = runner

    executor = AnthropicProcessExecutor()
    run_result = RunResult()
    state = IterationState()

    async def block_gen():
        yield SimpleNamespace(type="text", text="hi")
        yield SimpleNamespace(type="text", text="there")

    await executor._stream_blocks(process, block_gen(), run_result, state)
    if tasks:
        await asyncio.gather(*tasks)

    assert process.trigger_event.call_count >= 2
    process.trigger_event.assert_any_call(CallbackEvent.API_STREAM_BLOCK, block=SimpleNamespace(type="text", text="hi"))
    process.trigger_event.assert_any_call(
        CallbackEvent.API_STREAM_BLOCK, block=SimpleNamespace(type="text", text="there")
    )
