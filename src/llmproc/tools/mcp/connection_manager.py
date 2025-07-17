"""Manage connections to MCP servers."""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, get_default_environment, stdio_client

from .exceptions import MCPConnectionsDisabledError
from .persistent import _PersistentClient
from .server_registry import MCPServerSettings

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Handle persistent and transient MCP client connections."""

    def __init__(self, servers: dict[str, MCPServerSettings]) -> None:
        self.servers = servers
        self.transient = os.getenv("LLMPROC_MCP_TRANSIENT", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self._client_cms: dict[str, _PersistentClient] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

        def _close_all() -> None:  # pragma: no cover - teardown helper
            if self.transient or not self._client_cms:
                return
            loop = self._loop
            if loop is None or loop.is_closed():
                logger.debug("Skipping MCP client cleanup: no usable event loop")
                return
            try:
                if loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(self.close_clients(client_timeout=0.5), loop)
                    fut.result(timeout=2)
                else:
                    coro = asyncio.wait_for(self.close_clients(client_timeout=0.5), timeout=2)
                    loop.run_until_complete(coro)
            except TimeoutError:
                logger.warning("Timeout while closing MCP clients during exit")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to close MCP clients: %s", exc)

        atexit.register(_close_all)

    @asynccontextmanager
    async def get_client(self, server_name: str) -> AsyncGenerator[ClientSession, None]:
        """Yield a transient client connection to ``server_name``."""
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found in registry")
        config = self.servers[server_name]
        if config.type == "stdio":
            if not config.command or not config.args:
                raise ValueError(f"Command and args required for stdio type: {server_name}")
            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env={**get_default_environment(), **(config.env or {})},
            )
            async with stdio_client(params) as (read_stream, write_stream):
                session = ClientSession(read_stream, write_stream)
                async with session:
                    await session.initialize()
                    yield session
        elif config.type == "sse":
            if not config.url:
                raise ValueError(f"URL required for SSE type: {server_name}")
            async with sse_client(config.url) as (read_stream, write_stream):
                session = ClientSession(read_stream, write_stream)
                async with session:
                    await session.initialize()
                    yield session
        else:
            raise ValueError(f"Unsupported type: {config.type}")

    async def get_persistent_client(self, server_name: str) -> ClientSession:
        """Return a persistent client connection to ``server_name``."""
        if self.transient:
            raise MCPConnectionsDisabledError("Persistent MCP connections disabled; use get_client")
        if server_name in self._client_cms:
            return await self._client_cms[server_name].start()

        cm = self.get_client(server_name)
        client = _PersistentClient(cm)
        self._client_cms[server_name] = client
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return await client.start()

    async def close_clients(self, client_timeout: float = 1.0) -> None:  # pragma: no cover - API
        """Close all persistent clients."""
        if self.transient:
            return

        for server, client in list(self._client_cms.items()):
            try:
                await asyncio.wait_for(client.close(), timeout=client_timeout)
            except TimeoutError:
                logger.warning("Timeout closing MCP server '%s' after %s seconds", server, client_timeout)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Force-closing MCP server '%s' due to shutdown error: %s", server, exc)
            finally:
                self._client_cms.pop(server, None)
