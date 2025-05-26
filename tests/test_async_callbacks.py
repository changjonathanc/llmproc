import asyncio

import pytest

from llmproc.callbacks import CallbackEvent
from tests.conftest import create_test_llmprocess_directly


@pytest.mark.asyncio
async def test_async_callbacks_supported():
    """Callbacks can be async functions and methods."""
    process = create_test_llmprocess_directly()

    func_called = asyncio.Event()
    method_called = asyncio.Event()

    async def func_cb(event, *args, **kwargs):
        func_called.set()

    class ObjCb:
        async def response(self, text):
            method_called.set()

    process.add_callback(func_cb)
    process.add_callback(ObjCb())

    process.trigger_event(CallbackEvent.RESPONSE, "hi")

    await asyncio.wait_for(func_called.wait(), timeout=1)
    await asyncio.wait_for(method_called.wait(), timeout=1)


@pytest.mark.asyncio
async def test_mixed_sync_async_methods_supported():
    """Callback objects can mix async and sync methods."""
    process = create_test_llmprocess_directly()

    sync_called = asyncio.Event()
    async_called = asyncio.Event()

    class MixedCb:
        def tool_start(self, tool_name, tool_args):
            sync_called.set()

        async def response(self, text):
            async_called.set()

    process.add_callback(MixedCb())

    process.trigger_event(CallbackEvent.TOOL_START, "test", {})
    process.trigger_event(CallbackEvent.RESPONSE, "hi")

    await asyncio.wait_for(sync_called.wait(), timeout=1)
    await asyncio.wait_for(async_called.wait(), timeout=1)
