import asyncio

import pytest

from llmproc.plugin.events import CallbackEvent
from tests.conftest import create_test_llmprocess_directly


@pytest.mark.asyncio
async def test_async_callbacks_supported():
    """Callbacks can be async methods on objects."""
    process = create_test_llmprocess_directly()

    method_called = asyncio.Event()

    class ObjCb:
        async def hook_response(self, content, process):
            method_called.set()

    process.add_plugins(ObjCb())

    await process.plugins.response(process, "hi")

    await asyncio.wait_for(method_called.wait(), timeout=1)


@pytest.mark.asyncio
async def test_mixed_sync_async_methods_supported():
    """Callback objects can mix async and sync methods."""
    process = create_test_llmprocess_directly()

    sync_called = asyncio.Event()
    async_called = asyncio.Event()

    class MixedCb:
        def tool_start(self, tool_name, tool_args, *, process):
            sync_called.set()

        async def hook_response(self, content, process):
            async_called.set()

    process.add_plugins(MixedCb())

    await process.trigger_event(CallbackEvent.TOOL_START, tool_name="test", tool_args={})
    await process.plugins.response(process, "hi")

    await asyncio.wait_for(sync_called.wait(), timeout=1)
    await asyncio.wait_for(async_called.wait(), timeout=1)
