import asyncio
import threading
from typing import Optional


class EventLoopMixin:
    """Mixin providing private event loop management helpers."""

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = loop
        self._loop_thread: Optional[threading.Thread] = None
        self._own_loop: bool = False

    def _ensure_loop_thread(self) -> None:
        """Create and start a dedicated event loop thread if needed."""
        if self._loop is not None:
            return

        self._loop = asyncio.new_event_loop()
        self._own_loop = True

        def _runner() -> None:
            asyncio.set_event_loop(self._loop)  # type: ignore[arg-type]
            self._loop.run_forever()

        thread_name = f"EventLoopMixin-{id(self):x}"
        self._loop_thread = threading.Thread(target=_runner, name=thread_name, daemon=True)
        self._loop_thread.start()

    def _submit_to_loop(self, coro):  # type: ignore[valid-type]
        """Schedule ``coro`` on the mixin's loop and return a Future."""
        self._ensure_loop_thread()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _set_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        thread: Optional[threading.Thread],
        own: bool = False,
    ) -> None:
        """Assign an existing event loop and thread to the mixin."""
        self._loop = loop
        self._loop_thread = thread
        self._own_loop = own
