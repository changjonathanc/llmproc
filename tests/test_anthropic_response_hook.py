"""Tests for Anthropic executor response hook handling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import asyncio

import pytest

from llmproc.common.results import ToolResult
from llmproc.plugin.datatypes import ResponseHookResult
from llmproc.plugin.plugin_event_runner import PluginEventRunner
from llmproc.providers.anthropic_process_executor import AnthropicProcessExecutor


class _TextBlock(SimpleNamespace):
    def __init__(self, text: str):
        super().__init__(type="text", text=text)


class _FakeResponse(SimpleNamespace):
    def __init__(self, content):
        super().__init__(content=content, stop_reason="end_turn", usage={})


@pytest.fixture()
def minimal_process(monkeypatch):
    """Create a minimal process object for Anthropic executor tests."""
    proc = MagicMock()
    proc.model_name = "claude-3-haiku"
    proc.provider = "anthropic"
    proc.enriched_system_prompt = "system"
    proc.state = []
    proc.tool_manager = MagicMock(runtime_context={})
    proc.api_params = {}

    messages = MagicMock()
    proc.client = MagicMock(messages=messages)
    proc.trigger_event = AsyncMock()

    fd_plugin = MagicMock()
    fd_plugin.fd_manager = MagicMock(max_direct_output_chars=8000)
    proc.get_plugin = MagicMock(return_value=fd_plugin)

    proc.call_tool = AsyncMock(return_value=ToolResult.from_success("ok"))

    proc._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
    runner = PluginEventRunner(proc._submit_to_loop, [])
    proc.plugins = runner
    proc.hooks = runner

    class DummyRunResult:
        def __init__(self):
            self.api_calls = 0
            self.last_message = ""
            self.stop_reason = None

        def add_api_call(self, info):
            self.api_calls += 1

        def add_tool_call(self, tool_name, tool_args=None):
            pass

        def set_last_message(self, text):
            self.last_message = text
            return self

        def set_stop_reason(self, reason):
            self.stop_reason = reason
            return self

        def complete(self):
            return self

    monkeypatch.setattr(
        "llmproc.providers.anthropic_process_executor.RunResult",
        DummyRunResult,
    )

    return proc


@pytest.mark.asyncio
async def test_commit_current_false_discards(minimal_process):
    """Stopping with commit_current=False drops the assistant message."""

    plugin = MagicMock()
    plugin.hook_response = AsyncMock(
        return_value=ResponseHookResult(stop=True, commit_current=False)
    )

    runner = PluginEventRunner(minimal_process._submit_to_loop, [plugin])
    minimal_process.plugins = runner
    minimal_process.hooks = runner

    minimal_process.client.messages.create = AsyncMock(
        return_value=_FakeResponse([_TextBlock("hi")])
    )

    executor = AnthropicProcessExecutor()
    await executor.run(minimal_process, "hello", max_iterations=1)

    assert minimal_process.state[-1]["role"] == "user"


@pytest.mark.asyncio
async def test_commit_current_true_keeps(minimal_process):
    """Stopping with commit_current=True keeps the partial message."""

    plugin = MagicMock()
    plugin.hook_response = AsyncMock(
        return_value=ResponseHookResult(stop=True, commit_current=True)
    )

    runner = PluginEventRunner(minimal_process._submit_to_loop, [plugin])
    minimal_process.plugins = runner
    minimal_process.hooks = runner

    minimal_process.client.messages.create = AsyncMock(
        return_value=_FakeResponse([_TextBlock("hi")])
    )

    executor = AnthropicProcessExecutor()
    await executor.run(minimal_process, "hello", max_iterations=1)

    assert minimal_process.state[-1]["role"] == "assistant"
