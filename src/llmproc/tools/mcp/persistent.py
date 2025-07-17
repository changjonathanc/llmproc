"""Persistent client helper for MCP."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from mcp.client.session import ClientSession


class _PersistentClient:
    """Helper to keep a client session alive in a background task."""

    def __init__(self, cm: AsyncGenerator[ClientSession, None]):
        self._cm = cm
        self._task: asyncio.Task | None = None
        self._start = asyncio.Event()
        self._stop = asyncio.Event()
        self.session: ClientSession | None = None

    async def start(self) -> ClientSession:
        if self._task is None:
            self._task = asyncio.create_task(self._runner())
        await self._start.wait()
        assert self.session is not None
        return self.session

    async def _runner(self) -> None:
        async with self._cm as client:
            self.session = client
            self._start.set()
            await self._stop.wait()
        self.session = None

    async def close(self, timeout: float = 1.0) -> None:
        if self._task is None:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        except TimeoutError:
            pass
