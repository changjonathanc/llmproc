"""Tests for the write_stderr builtin tool."""

import pytest
from typing import Any
from llmproc.common.results import ToolResult
from llmproc.callbacks import CallbackEvent
from llmproc.tools.builtin.write_stderr import write_stderr_tool


@pytest.mark.asyncio
async def test_write_stderr_list_buffer():
    """write_stderr_tool appends to list buffer and triggers event."""

    class DummyProcess:
        def __init__(self) -> None:
            self.events: list[tuple[Any, str]] = []

        def trigger_event(self, event: CallbackEvent, message: str) -> None:
            self.events.append((event, message))

    proc = DummyProcess()
    buffer: list[str] = []
    result = await write_stderr_tool(
        "hello", runtime_context={"process": proc, "stderr": buffer}
    )
    assert isinstance(result, ToolResult)
    assert not result.is_error
    assert buffer == ["hello"]
    assert proc.events == [(CallbackEvent.STDERR_WRITE, "hello")]


@pytest.mark.asyncio
async def test_write_stderr_missing_context():
    """Missing stderr key triggers error."""
    result = await write_stderr_tool("oops", runtime_context={})
    assert isinstance(result, ToolResult)
    assert result.is_error
