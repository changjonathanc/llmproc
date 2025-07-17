"""Tests for automatic ToolResult wrapping in :class:`Tool.execute`."""

import pytest

from llmproc.common.results import ToolResult
from llmproc.tools.tool_registry import ToolRegistry
from llmproc.tools.core import Tool
from llmproc.common.metadata import get_tool_meta


@pytest.mark.asyncio
async def test_call_tool_wraps_string_result():
    """Ensure raw string results are converted to ToolResult."""
    registry = ToolRegistry()

    async def hello_tool(**kwargs):
        return "hello"

    registry.register_tool_obj(
        Tool(
            handler=hello_tool,
            schema={"name": "hello", "description": "test", "input_schema": {"type": "object", "properties": {}}},
            meta=get_tool_meta(hello_tool),
        )
    )

    result = await registry.get_tool("hello").execute({}, runtime_context=None)
    assert isinstance(result, ToolResult)
    assert result.content == "hello"
    assert not result.is_error


@pytest.mark.asyncio
async def test_call_tool_wraps_dict_result():
    """Ensure dictionary results are converted to ToolResult."""
    registry = ToolRegistry()

    async def dict_tool(**kwargs):
        return {"a": 1}

    registry.register_tool_obj(
        Tool(
            handler=dict_tool,
            schema={"name": "dict", "description": "test", "input_schema": {"type": "object", "properties": {}}},
            meta=get_tool_meta(dict_tool),
        )
    )

    result = await registry.get_tool("dict").execute({}, runtime_context=None)
    assert isinstance(result, ToolResult)
    assert result.content == {"a": 1}
    assert not result.is_error
