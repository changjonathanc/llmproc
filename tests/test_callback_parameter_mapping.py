"""Tests for correct callback parameter mapping in trigger_event."""

from typing import Any
import pytest

from llmproc.plugin.events import CallbackEvent
from tests.conftest import create_test_llmprocess_directly


@pytest.mark.asyncio
async def test_tool_start_and_response_parameters():
    """Callback methods should receive correctly mapped parameters."""
    process = create_test_llmprocess_directly()

    calls: dict[str, Any] = {}

    class MyCallback:
        def tool_start(self, tool_name: str, tool_args: Any, *, process) -> None:
            calls["tool_name"] = tool_name
            calls["args"] = tool_args

        def hook_response(self, content: str, process) -> None:
            calls["content"] = content

    process.add_plugins(MyCallback())
    await process.trigger_event(CallbackEvent.TOOL_START, tool_name="calc", tool_args={"x": 1})
    await process.plugins.response(process, "done")

    assert calls["tool_name"] == "calc"
    assert calls["args"] == {"x": 1}
    assert calls["content"] == "done"


@pytest.mark.asyncio
async def test_api_stream_block_parameter():
    """api_stream_block should pass the block object unmodified."""
    process = create_test_llmprocess_directly()

    captured: dict[str, Any] = {}

    class MyCb:
        def api_stream_block(self, block) -> None:
            captured["block"] = block

    process.add_plugins(MyCb())
    blk = {"type": "text", "text": "hi"}
    await process.trigger_event(CallbackEvent.API_STREAM_BLOCK, block=blk)

    assert captured["block"] is blk
