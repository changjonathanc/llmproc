"""Load tool definitions from MCP servers."""

from __future__ import annotations

import asyncio
import logging
import os
from asyncio import gather
from collections.abc import AsyncGenerator, Awaitable, Callable

from mcp.client.session import ClientSession
from mcp.types import ListToolsResult
from mcp.types import Tool as MCPTool

from .connection_manager import ConnectionManager
from .constants import MCP_DEFAULT_TOOL_FETCH_TIMEOUT
from .exceptions import MCPServerConnectionError
from .namespaced_tool import NamespacedTool
from .server_registry import MCPServerSettings

logger = logging.getLogger(__name__)


class ToolLoader:
    """Responsible for loading tool definitions from MCP servers."""

    def __init__(
        self,
        servers: dict[str, MCPServerSettings],
        connection_manager: ConnectionManager,
        tool_filter: dict[str, list[str] | None] | None = None,
        separator: str = "__",
        client_factory: Callable[[str], AsyncGenerator[ClientSession, None]] | None = None,
        persistent_client_factory: Callable[[str], Awaitable[ClientSession]] | None = None,
    ) -> None:
        self.servers = servers
        self.connection_manager = connection_manager
        self.client_factory = client_factory or connection_manager.get_client
        self.persistent_client_factory = persistent_client_factory or connection_manager.get_persistent_client
        self.separator = separator
        self.tool_filter = tool_filter or {}
        self._namespaced_tools: dict[str, NamespacedTool] = {}

    async def _load_server_tools(self, server_name: str) -> tuple[str, list[MCPTool]]:
        timeout = float(os.environ.get("LLMPROC_TOOL_FETCH_TIMEOUT", MCP_DEFAULT_TOOL_FETCH_TIMEOUT))
        try:
            async with asyncio.timeout(timeout):
                if self.connection_manager.transient:
                    async with self.client_factory(server_name) as client:
                        result: ListToolsResult = await client.list_tools()
                else:
                    client = await self.persistent_client_factory(server_name)
                    result = await client.list_tools()
                tools = result.tools or []
                logger.debug("Loaded %s tools from %s", len(tools), server_name)
                return server_name, tools
        except Exception as exc:  # noqa: BLE001
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status:
                msg = f"Server '{server_name}' unreachable (status {status}): {exc}"
            else:
                msg = f"Server '{server_name}' unreachable: {exc}"
            logger.error(msg)
            raise MCPServerConnectionError(msg) from exc

    def _should_include_tool(self, server_name: str, tool_name: str) -> bool:
        if server_name not in self.tool_filter:
            return True
        tool_list = self.tool_filter[server_name]
        if tool_list is None:
            return True
        if not tool_list:
            return False
        is_negative = tool_list[0].startswith("-")
        if is_negative:
            return not any(t[1:] == tool_name for t in tool_list)
        return tool_name in tool_list

    async def load_servers(self, specific_servers: list[str] | None = None) -> None:
        servers_to_load = specific_servers or list(self.servers.keys())
        if len(servers_to_load) > 1:
            logger.debug("Loading tools from servers: %s", servers_to_load)
        elif len(servers_to_load) == 1:
            logger.debug("Loading tools from server: %s", servers_to_load[0])
        else:
            logger.debug("No servers to load")
            return

        if specific_servers:
            for name in list(self._namespaced_tools.keys()):
                nt = self._namespaced_tools[name]
                if nt.server_name in specific_servers:
                    del self._namespaced_tools[name]
        else:
            self._namespaced_tools.clear()

        results = await gather(*(self._load_server_tools(name) for name in servers_to_load))

        for server_name, tools in results:
            for tool in tools:
                original_name = tool.name
                if not self._should_include_tool(server_name, original_name):
                    continue

                namespaced_name = f"{server_name}{self.separator}{original_name}"
                namespaced_tool = tool.model_copy(update={"name": namespaced_name})
                namespaced_tool.description = f"[{server_name}] {tool.description or ''}"
                self._namespaced_tools[namespaced_name] = NamespacedTool(
                    tool=namespaced_tool,
                    server_name=server_name,
                    original_name=original_name,
                )

    def list_tools(self) -> ListToolsResult:
        tools = [nt.tool for nt in self._namespaced_tools.values()]
        result_dict = {"tools": []}
        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "inputSchema"):
                tool_dict = {"name": tool.name, "inputSchema": tool.inputSchema}
                if hasattr(tool, "description") and tool.description:
                    tool_dict["description"] = tool.description
                result_dict["tools"].append(tool_dict)
        return ListToolsResult(**result_dict)

    def get_namespaced_tools(self) -> dict[str, NamespacedTool]:
        return self._namespaced_tools
