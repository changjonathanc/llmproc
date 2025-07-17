"""MCP Aggregator module."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ListToolsResult,
    TextContent,
)

from llmproc.common.metadata import ToolMeta, attach_meta
from llmproc.common.results import ToolResult
from llmproc.config.mcp import MCPServerTools
from llmproc.tools.core import Tool
from llmproc.tools.function_schemas import create_schema_from_callable
from llmproc.tools.mcp.connection_manager import ConnectionManager
from llmproc.tools.mcp.constants import (
    MCP_DEFAULT_TOOL_CALL_TIMEOUT,
    MCP_ERROR_TOOL_CALL_TIMEOUT,
)
from llmproc.tools.mcp.exceptions import (
    MCPConnectionsDisabledError,
    MCPServerConnectionError,
    MCPToolsLoadingError,
)
from llmproc.tools.mcp.namespaced_tool import NamespacedTool
from llmproc.tools.mcp.persistent import _PersistentClient
from llmproc.tools.mcp.server_registry import MCPServerSettings
from llmproc.tools.mcp.tool_loader import ToolLoader

logger = logging.getLogger(__name__)


def _error_result(message: str) -> CallToolResult:
    """Return a simple error result."""
    return CallToolResult(isError=True, message=message, content=[TextContent(type="text", text=message)])


def _split_tool_name(tool: str, server: str | None, sep: str) -> tuple[str, str] | None:
    """Return ``(server, tool)`` from a possibly namespaced tool name."""
    if server:
        return server, tool
    if sep not in tool:
        return None
    return tuple(tool.split(sep, 1))  # type: ignore[return-value]


def _extract_content(result) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Normalize various result formats into a list of content objects."""
    content: list[TextContent | ImageContent | EmbeddedResource] = []
    extracted = None
    if hasattr(result, "content"):
        extracted = result.content
    elif isinstance(result, dict) and "content" in result:
        extracted = result["content"]
    elif hasattr(result, "result"):
        extracted = [result.result]
    elif isinstance(result, dict) and "result" in result:
        extracted = [result["result"]]

    if extracted:
        for item in extracted:
            if isinstance(item, TextContent | ImageContent | EmbeddedResource):
                content.append(item)
            elif isinstance(item, dict) and "text" in item and "type" in item:
                content.append(TextContent(**item))
            elif isinstance(item, str):
                content.append(TextContent(type="text", text=item))
            else:
                content.append(TextContent(type="text", text=str(item)))
    return content


def _process_call_result(result, server: str, tool: str) -> CallToolResult:
    """Convert a raw MCP response into a :class:`CallToolResult`."""
    if getattr(result, "isError", False):
        error_message = getattr(result, "message", "")
        error_content = getattr(result, "content", [])

        detailed = f"MCP server '{server}' returned error for tool '{tool}'"
        if error_message:
            detailed += f": {error_message}"
        if error_content:
            texts = []
            for item in error_content:
                if hasattr(item, "text"):
                    texts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
            if texts:
                detailed += f" | Content: {' | '.join(texts)}"
        if not error_message and not error_content:
            attrs = [a for a in dir(result) if not a.startswith("_")]
            detailed += f" | Available attributes: {attrs}"
        logger.error(detailed)

        simple = f"Server '{server}' returned error"
        if error_message:
            simple += f": {error_message}"
        return _error_result(simple)

    content = _extract_content(result)
    if not content:
        content = [TextContent(type="text", text="Tool execution completed.")]
    return CallToolResult(isError=False, message="", content=content)


def create_mcp_tool_handler(aggregator: MCPAggregator, server_name: str, tool_name: str) -> Callable:
    """Return a bound handler for an MCP tool call."""

    async def tool_handler(**kwargs) -> ToolResult:
        try:
            result = await aggregator.call_tool_resolved(server_name, tool_name, kwargs)
            if result.isError:
                return ToolResult(content=result.content, is_error=True)
            return ToolResult(content=result.content, is_error=False)
        except Exception as exc:  # noqa: BLE001
            error_message = f"Error calling MCP tool {server_name}__{tool_name}: {exc}"
            logger.error(error_message)
            return ToolResult.from_error(error_message)

    return tool_handler


class MCPAggregator:
    """Aggregate multiple MCP servers into a single interface."""

    def __init__(
        self,
        servers: dict[str, MCPServerSettings],
        tool_filter: dict[str, list[str] | None] | None = None,
        separator: str = "__",
    ) -> None:
        self.servers = servers

        if tool_filter is not None:
            for server, tools in tool_filter.items():
                if tools is not None and not isinstance(tools, list):
                    raise ValueError(
                        f"Invalid tool_filter for server '{server}': value must be a list or None, got {type(tools).__name__}"
                    )
                if (
                    tools is not None
                    and any(t.startswith("-") for t in tools)
                    and any(not t.startswith("-") for t in tools)
                ):
                    raise ValueError(
                        f"Mixed filter types for server '{server}'. Use either all positive filters or all negative filters."
                    )
            self.tool_filter = tool_filter
        else:
            self.tool_filter = {}

        self.separator = separator
        self.connection_manager = ConnectionManager(servers)
        self.loader = ToolLoader(
            servers,
            self.connection_manager,
            self.tool_filter,
            separator,
            client_factory=self.get_client,
            persistent_client_factory=self._get_or_create_client,
        )

    @property
    def _namespaced_tools(self) -> dict[str, NamespacedTool]:
        return self.loader.get_namespaced_tools()

    @_namespaced_tools.setter
    def _namespaced_tools(self, value: dict[str, NamespacedTool]) -> None:
        self.loader._namespaced_tools = value

    @classmethod
    def from_dict(
        cls,
        config: dict,
        tool_filter: dict[str, list[str] | None] | None = None,
        separator: str = "__",
    ) -> MCPAggregator:
        servers = {name: MCPServerSettings(**settings) for name, settings in config.items()}
        return cls(servers, tool_filter=tool_filter, separator=separator)

    @classmethod
    def from_config(
        cls,
        path: Path | str,
        tool_filter: dict[str, list[str] | None] | None = None,
        separator: str = "__",
    ) -> MCPAggregator:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            config = json.load(f)
        if "mcpServers" not in config:
            raise KeyError("Config file must have a 'mcpServers' section")
        servers = {name: MCPServerSettings(**settings) for name, settings in config["mcpServers"].items()}
        return cls(servers, tool_filter=tool_filter, separator=separator)

    def filter_servers(self, server_names: list[str]) -> MCPAggregator:
        missing = [name for name in server_names if name not in self.servers]
        if missing:
            available = ", ".join(self.servers.keys())
            raise ValueError(f"Servers not found: {', '.join(missing)}. Available servers: {available}")
        filtered = {name: self.servers[name] for name in server_names}
        return MCPAggregator(filtered, tool_filter=self.tool_filter, separator=self.separator)

    @asynccontextmanager
    async def get_client(self, server_name: str) -> AsyncGenerator[ClientSession, None]:
        async with self.connection_manager.get_client(server_name) as client:
            yield client

    async def _get_or_create_client(self, server_name: str) -> ClientSession:
        cm_mgr = self.connection_manager
        if cm_mgr.transient:
            raise MCPConnectionsDisabledError("Persistent MCP connections disabled; use get_client")
        if server_name in cm_mgr._client_cms:
            return await cm_mgr._client_cms[server_name].start()

        cm = self.get_client(server_name)
        client = _PersistentClient(cm)
        cm_mgr._client_cms[server_name] = client
        if cm_mgr._loop is None:
            cm_mgr._loop = asyncio.get_running_loop()
        return await client.start()

    async def close_clients(self, client_timeout: float = 1.0) -> None:  # pragma: no cover - API
        await self.connection_manager.close_clients(client_timeout=client_timeout)

    async def load_servers(self, specific_servers: list[str] | None = None) -> None:
        await self.loader.load_servers(specific_servers)

    async def list_tools(self) -> ListToolsResult:
        """Return tool definitions from all configured servers."""
        try:
            await self.load_servers()
        except Exception as exc:
            msg = f"Error loading servers: {exc}"
            logger.error(msg)
            raise MCPToolsLoadingError(msg) from exc

        return self.loader.list_tools()

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict | None = None,
        server_name: str | None = None,
    ) -> CallToolResult:
        names = _split_tool_name(tool_name, server_name, self.separator)
        if not names:
            err_msg = f"Tool name '{tool_name}' must be namespaced as 'server{self.separator}tool'"
            return _error_result(err_msg)
        actual_server, actual_tool = names
        return await self.call_tool_resolved(actual_server, actual_tool, arguments)

    async def call_tool_resolved(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict | None = None,
    ) -> CallToolResult:
        actual_server = server_name
        actual_tool = tool_name

        if actual_server not in self.servers:
            available = ", ".join(self.servers.keys())
            return _error_result(f"Server '{actual_server}' not found in registry. Available servers: {available}")

        try:
            await self.load_servers(specific_servers=[actual_server])
        except Exception as exc:
            err_msg = f"Error loading server '{actual_server}': {exc}"
            logger.error(err_msg)
            return _error_result(err_msg)

        namespaced_tool_name = f"{actual_server}{self.separator}{actual_tool}"
        if namespaced_tool_name not in self.loader.get_namespaced_tools():
            if (
                actual_server in self.tool_filter
                and self.tool_filter[actual_server] is not None
                and actual_tool not in self.tool_filter[actual_server]
            ):
                return _error_result(f"Tool '{actual_tool}' not found or filtered out from server '{actual_server}'")

        def process_result(result) -> CallToolResult:
            return _process_call_result(result, actual_server, actual_tool)

        try:
            tool_call_timeout = float(os.environ.get("LLMPROC_TOOL_CALL_TIMEOUT", MCP_DEFAULT_TOOL_CALL_TIMEOUT))
            if self.connection_manager.transient:
                async with self.get_client(actual_server) as client:
                    async with asyncio.timeout(tool_call_timeout):
                        result = await client.call_tool(actual_tool, arguments)
                        return process_result(result)
            else:
                client = await self._get_or_create_client(actual_server)
                async with asyncio.timeout(tool_call_timeout):
                    result = await client.call_tool(actual_tool, arguments)
                    return process_result(result)
        except TimeoutError:
            cfg = self.servers[actual_server]
            server_info = f"Server type: {cfg.type}"
            if cfg.type == "sse":
                server_info += f", URL: {cfg.url}"
            elif cfg.type == "stdio":
                server_info += f", Command: {cfg.command}"
            tool_call_timeout = float(os.environ.get("LLMPROC_TOOL_CALL_TIMEOUT", MCP_DEFAULT_TOOL_CALL_TIMEOUT))
            err_msg = MCP_ERROR_TOOL_CALL_TIMEOUT.format(
                tool=actual_tool, server=actual_server, timeout=tool_call_timeout
            )
            err_msg += f" {server_info}"
            logger.error(err_msg)
            return _error_result(err_msg)
        except Exception as e:  # noqa: BLE001
            err_msg = f"Error in call_tool for '{tool_name}': {e}"
            logger.error(err_msg)
            return _error_result(err_msg)

    async def initialize(
        self,
        descriptors: list[MCPServerTools],
        config: dict[str, Any] | None = None,
    ) -> list[Tool]:
        """Load servers and return initialized :class:`Tool` objects."""
        config = config or {}
        await self.load_servers()

        regs: list[Tool] = []
        for desc in descriptors:
            for nt in self.loader.get_namespaced_tools().values():
                if nt.server_name != desc.server or not self._is_allowed(desc, nt.original_name):
                    continue
                regs.append(self._create_tool(nt, desc, config))
        return regs

    @staticmethod
    def _is_allowed(desc: MCPServerTools, name: str) -> bool:
        if desc.tools == "all":
            return True
        for item in desc.tools:
            if isinstance(item, str) and item == name:
                return True
            if not isinstance(item, str) and getattr(item, "name", None) == name:
                return True
        return False

    def _create_tool(self, nt: NamespacedTool, desc: MCPServerTools, config: dict[str, Any]) -> Tool:
        namespaced_tool_name = nt.tool.name
        handler = create_mcp_tool_handler(self, nt.server_name, nt.original_name)
        access_level = desc.get_access_level(nt.original_name)
        cfg = desc._find_tool(nt.original_name)
        override_desc = cfg.description if cfg is not None else None
        param_desc = cfg.param_descriptions if cfg is not None else None
        public_name = cfg.alias if cfg is not None else namespaced_tool_name

        existing_desc: dict[str, str] = {}
        for pname, prop in (nt.tool.inputSchema or {}).get("properties", {}).items():
            if isinstance(prop, dict) and "description" in prop:
                existing_desc[pname] = prop["description"]
        if param_desc:
            existing_desc.update(param_desc)

        meta = ToolMeta(
            name=public_name,
            access=access_level,
            description=override_desc or nt.tool.description,
            param_descriptions=existing_desc or None,
            raw_schema={
                "name": namespaced_tool_name,
                "description": nt.tool.description,
                "input_schema": nt.tool.inputSchema or {"type": "object", "properties": {}},
            },
        )
        attach_meta(handler, meta)
        schema = create_schema_from_callable(handler, config)
        return Tool(handler=handler, schema=schema, meta=meta)
