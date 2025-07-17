"""Tests for the write_stderr builtin tool."""

import pytest
from typing import Any
from llmproc.common.results import ToolResult
from llmproc.plugins.stderr import StderrPlugin
from llmproc.tools.core import Tool


@pytest.mark.asyncio
async def test_write_stderr_list_buffer():
    """write_stderr_tool appends to list buffer."""

    class DummyProcess:
        def __init__(self) -> None:
            self.events: list[tuple[Any, str]] = []

        def trigger_event(self, event: Any, message: str) -> None:
            self.events.append((event, message))

    proc = DummyProcess()
    plugin = StderrPlugin()
    tool = Tool.from_callable(plugin.write_stderr_tool)
    result = await tool.execute(
        {"message": "hello"},
        runtime_context={"process": proc},
    )
    assert isinstance(result, ToolResult)
    assert not result.is_error
    assert plugin.log == ["hello"]
    # No custom STDERR_WRITE event is triggered
    assert proc.events == []



@pytest.mark.asyncio
async def test_write_stderr_missing_context():
    """Missing process key triggers error."""
    plugin = StderrPlugin()
    tool = Tool.from_callable(plugin.write_stderr_tool)
    result = await tool.execute({"message": "oops"}, runtime_context={})
    assert isinstance(result, ToolResult)
    assert result.is_error
