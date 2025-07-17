"""Utilities for running async iterables in the background."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class AsyncBackgroundIterator(AsyncIterator[T], Generic[T]):
    """Run an async iterable in the background and yield its items."""

    def __init__(self, agen: AsyncIterable[T], on_item: Callable[[T], None] | None = None):
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._agen = agen
        self._on_item = on_item
        self._task: asyncio.Task[Any] | None = None
        self._closed = False

    async def __aenter__(self) -> AsyncBackgroundIterator[T]:
        """Start background task and return the iterator."""
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Terminate the background task on exit."""
        await self.aclose()

    async def _run(self) -> None:
        try:
            async for item in self._agen:
                print(item, flush=True)
                if self._on_item is not None:
                    try:
                        result = self._on_item(item)
                        if inspect.iscoroutine(result):
                            await result
                    except Exception:  # pragma: no cover - swallow callback errors
                        pass
                await self._queue.put(item)
        except Exception as e:  # pragma: no cover - pass through errors
            await self._queue.put(e)
        finally:
            await self._queue.put(StopAsyncIteration)

    def __aiter__(self) -> AsyncBackgroundIterator[T]:
        """Return the iterator itself."""
        return self

    async def __anext__(self) -> T:
        """Yield the next item from the background task."""
        item = await self._queue.get()
        if isinstance(item, Exception):
            raise item
        if item is StopAsyncIteration:
            self._closed = True
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        """Cancel the background task if it is still running."""
        if not self._closed:
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:  # pragma: no cover - task cancelled
                    pass
            self._closed = True
