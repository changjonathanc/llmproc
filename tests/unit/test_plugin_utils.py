"""Tests for plugin utility helpers."""

import pytest

from llmproc.plugin.plugin_utils import call_plugin, call_plugin_async, has_plugin_method


class _Plugin:
    def greet(self, name: str) -> str:
        return f"hi {name}"


class _AsyncPlugin:
    async def greet(self, name: str) -> str:
        return f"hi {name}"


async def _func_plugin(event_type: str, data: str):
    if event_type == "greet":
        return f"hello {data}"
    return None


def test_has_plugin_method():
    """has_plugin_method detects methods on objects."""
    plugin = _Plugin()
    assert has_plugin_method(plugin, "greet")
    assert not has_plugin_method(plugin, "missing")


@pytest.mark.asyncio
async def test_call_plugin_sync_method():
    """call_plugin works with synchronous methods."""
    plugin = _Plugin()
    result = await call_plugin_async(plugin, "greet", "claude")
    assert result == "hi claude"


@pytest.mark.asyncio
async def test_call_plugin_async_method():
    """call_plugin works with async methods."""
    plugin = _AsyncPlugin()
    result = await call_plugin_async(plugin, "greet", "claude")
    assert result == "hi claude"


@pytest.mark.asyncio
async def test_call_plugin_function():
    """call_plugin treats plain functions as plugins."""
    result = await call_plugin_async(_func_plugin, "greet", "world", function_first_arg="greet")
    assert result == "hello world"
