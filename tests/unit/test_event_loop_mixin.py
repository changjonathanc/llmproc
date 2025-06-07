"""Unit tests for EventLoopMixin."""

import asyncio
import threading

from llmproc.event_loop_mixin import EventLoopMixin
from tests.conftest import create_test_llmprocess_directly


class Dummy(EventLoopMixin):
    """Simple subclass for testing the mixin standalone."""

    def __init__(self, loop=None):
        EventLoopMixin.__init__(self, loop)


def test_mixin_starts_loop_and_runs_coroutine():
    """Ensure _submit_to_loop creates a loop and executes the coroutine."""
    dummy = Dummy()

    async def add(a, b):
        return a + b

    fut = dummy._submit_to_loop(add(1, 2))
    assert fut.result(timeout=1) == 3
    assert dummy._loop is not None
    assert dummy._own_loop
    assert dummy._loop_thread is not None

    dummy._loop.call_soon_threadsafe(dummy._loop.stop)
    dummy._loop_thread.join(timeout=1)


def test_mixin_set_loop_assigns_existing_loop():
    """_set_loop should use a provided loop without starting a new one."""
    loop = asyncio.new_event_loop()

    def runner():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()

    dummy = Dummy()
    dummy._set_loop(loop, thread)

    async def sample():
        return "ok"

    fut = dummy._submit_to_loop(sample())
    assert fut.result(timeout=1) == "ok"
    assert dummy._loop is loop
    assert dummy._loop_thread is thread
    assert not dummy._own_loop

    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)


def test_llmprocess_uses_event_loop_mixin():
    """LLMProcess should create a loop on demand via the mixin."""
    process = create_test_llmprocess_directly()

    async def value():
        return 5

    fut = process._submit_to_loop(value())
    assert fut.result(timeout=1) == 5
    assert process._loop is not None
    assert process._own_loop
    assert process._loop_thread is not None

    process._loop.call_soon_threadsafe(process._loop.stop)
    process._loop_thread.join(timeout=1)
