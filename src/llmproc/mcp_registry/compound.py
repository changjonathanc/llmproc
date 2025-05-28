"""
Core functionality for managing and aggregating MCP servers.

This module provides the main components for managing server configurations and
aggregating multiple MCP servers into a single interface. It includes:

- MCPServerSettings: Configuration settings for individual MCP servers
- ServerRegistry: Registry for managing multiple server configurations
 - MCPAggregator: Aggregator for combining multiple servers into a single interface
"""

import asyncio
import atexit
import json
import logging
import os
import sys
from asyncio import gather
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import (
    StdioServerParameters,
    get_default_environment,
    stdio_client,
)
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ListToolsResult,
    TextContent,
    Tool,
)
from pydantic import BaseModel

from llmproc.tools.mcp.constants import (
    MCP_DEFAULT_TOOL_CALL_TIMEOUT,
    MCP_ERROR_TOOL_CALL_TIMEOUT,
)

# Set up logging for debugging purposes.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class _PersistentClient:
    """Helper to keep a client session alive in a background task."""

    def __init__(self, cm: AsyncGenerator[ClientSession, None]):
        self._cm = cm
        self._task: asyncio.Task | None = None
        self._start = asyncio.Event()
        self._stop = asyncio.Event()
        self.session: ClientSession | None = None

    async def start(self) -> ClientSession:
        """Start the background context manager."""
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
        """Signal the background task to exit.

        Args:
            timeout: Maximum time in seconds to wait for the task to complete (default: 1.0)
        """
        if self._task is None:
            return
        self._stop.set()
        try:
            # Use asyncio.wait_for to add timeout
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        except TimeoutError:
            # If task doesn't complete in time, just continue
            pass


class MCPServerSettings(BaseModel):
    """
    Configuration settings for an individual MCP server.

    This class defines the settings needed to connect to and interact with an MCP server.
    It supports both stdio and SSE transport types.

    Attributes:
        type: Transport type ("stdio" or "sse")
        command: Command to run for stdio servers
        args: Command arguments for stdio servers
        url: URL for sse servers
        env: Environment variables to set for the server process
        description: Optional description of the server
    """

    type: str = "stdio"  # "stdio" or "sse"
    command: str | None = None  # for stdio
    args: list[str] | None = None  # for stdio
    url: str | None = None  # for sse
    env: dict | None = None
    description: str | None = None

    @property
    def transport(self) -> str:
        """
        Get the transport type.

        Returns:
            str: The transport type ("stdio" or "sse")
        """
        return self.type

    @transport.setter
    def transport(self, value: str) -> None:
        """
        Set the transport type.

        Args:
            value: The new transport type ("stdio" or "sse")
        """
        self.type = value


class ServerRegistry:
    """
    Registry for managing MCP server configurations.

    This class provides functionality to manage multiple server configurations,
    including loading from and saving to a configuration file, creating client
    sessions for servers, and retrieving server information.

    Attributes:
        registry: Dictionary mapping server names to their configuration settings
    """

    def __init__(self, servers: dict[str, MCPServerSettings]):
        """
        Initialize a ServerRegistry with a dictionary of server configurations.

        Args:
            servers: Dictionary mapping server names to their configuration settings
        """
        self.registry = servers

    def filter_servers(self, server_names: list[str]) -> "ServerRegistry":
        """
        Create a new ServerRegistry containing only the specified servers.

        Args:
            server_names: List of server names to include in the filtered registry

        Returns:
            ServerRegistry: A new registry containing only the specified servers

        Raises:
            ValueError: If any of the specified servers are not in the registry
        """
        missing = [name for name in server_names if name not in self.registry]
        if missing:
            raise ValueError(f"Servers not found: {', '.join(missing)}")

        filtered = {name: settings for name, settings in self.registry.items() if name in server_names}
        return ServerRegistry(filtered)

    @classmethod
    def from_dict(cls, config: dict) -> "ServerRegistry":
        """
        Create a ServerRegistry from a dictionary of server configurations.

        Args:
            config: A dictionary where keys are server names and values are server configuration dictionaries.
                   Each server configuration should contain fields matching MCPServerSettings.

        Returns:
            ServerRegistry: A new registry instance with the configured servers.

        Example:
            config = {
                "server1": {
                    "type": "stdio",
                    "command": "python",
                    "args": ["-m", "server"],
                    "description": "Python server"
                },
                "server2": {
                    "type": "sse",
                    "url": "http://localhost:8000/sse",
                    "description": "SSE server"
                }
            }
            registry = ServerRegistry.from_dict(config)
        """
        servers = {name: MCPServerSettings(**settings) for name, settings in config.items()}
        return cls(servers)

    @classmethod
    def from_config(cls, path: Path | str) -> "ServerRegistry":
        """
        Load a ServerRegistry from a configuration file.

        This method loads server configurations from a JSON file and creates
        a new ServerRegistry instance with those configurations. The file must
        contain a 'mcpServers' section with server configurations.

        Args:
            path: Path to the configuration file (JSON format)
                (can be a string or Path object)

        Returns:
            ServerRegistry: A new registry instance with the configured servers

        Raises:
            FileNotFoundError: If the configuration file does not exist
            KeyError: If the configuration file does not have a 'mcpServers' section
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            config = json.load(f)
        if "mcpServers" not in config:
            raise KeyError("Config file must have a 'mcpServers' section")
        servers = {name: MCPServerSettings(**settings) for name, settings in config["mcpServers"].items()}
        return cls(servers)

    @asynccontextmanager
    async def get_client(self, server_name: str) -> AsyncGenerator[ClientSession, None]:
        """
        Get a client session for a specific server.

        This async context manager creates a temporary connection to the specified
        server and yields a client session that can be used to interact with it.
        The connection is automatically closed when exiting the context.

        Args:
            server_name: Name of the server to connect to

        Yields:
            ClientSession: A session connected to the specified server

        Raises:
            ValueError: If the server is not found in the registry or has invalid configuration

        Example:
            ```python
            registry = ServerRegistry.from_config("config.json")
            async with registry.get_client("memory") as client:
                result = await client.call_tool("get", {"key": "test"})
            ```
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry")
        config = self.registry[server_name]
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

    def list_servers(self) -> list[str]:
        """
        Get a list of all server names in the registry.

        Returns:
            list[str]: List of server names
        """
        return list(self.registry.keys())


class NamespacedTool(BaseModel):
    """
    A tool that is namespaced by server name.

    This class represents a tool from a specific server, with its name
    prefixed by the server name (e.g., "server_name__tool_name").

    Attributes:
        tool: The tool object with its name updated to include the server prefix
        server_name: Name of the server that provides this tool
        namespaced_tool_name: The tool name with server prefix
        original_name: The original tool name without server prefix
    """

    tool: Tool
    server_name: str
    namespaced_tool_name: str
    original_name: str


class MCPAggregator:
    """
    Aggregates multiple MCP servers into a single interface.

    This class allows you to interact with multiple MCP servers through a unified interface,
    with tools from different servers namespaced by their server name. All tool calls
    use temporary connections created on demand.

    Examples:
        ```python
        # Method 1: Temporary connections (default behavior)
        aggregator = MCPAggregator(registry)
        result = await aggregator.call_tool("memory__get", {"key": "test"})

        # Method 2: Filtered registry - only include specific servers
        filtered_registry = registry.filter_servers(["memory", "github"])
        aggregator = MCPAggregator(filtered_registry)
        result = await aggregator.call_tool("memory__get", {"key": "test"})

        # Method 3: Filtered tools - only expose certain tools from servers
        tool_filter = {
            "memory": ["get", "set"],  # Only include get/set from memory
            "github": ["list_repos", "create_issue"],  # Only specific github tools
        }
        aggregator = MCPAggregator(registry, tool_filter=tool_filter)
        result = await aggregator.call_tool("memory__get", {"key": "test"})

        # Method 4: Reuse the aggregator for multiple calls
        aggregator = MCPAggregator(registry)
        result1 = await aggregator.call_tool("memory__get", {"key": "test"})
        result2 = await aggregator.call_tool("memory__set", {"key": "test", "value": "hello"})
        ```

    Attributes:
        registry: The ServerRegistry containing server configurations
        server_names: List of all server names in the registry (convenience reference)
        tool_filter: Dictionary mapping server names to lists of tool names to include
        separator: Character(s) used to separate server name from tool name
        _namespaced_tool_map: Internal mapping of namespaced tool names to tool information
    """

    def __init__(
        self,
        registry: ServerRegistry,
        tool_filter: dict[str, list[str] | None] | None = None,
        separator: str = "__",
    ):
        """
        Initialize the aggregator.

        Args:
            registry: ServerRegistry containing server configurations
            tool_filter: Optional dict mapping server names to lists of tool names to include.
                        If a server is mapped to None, all tools from that server are included.
                        If a server is not in the dict, all tools from that server are included.
                        If tool names start with "-", they are excluded (negative filtering).
            separator: Separator string between server name and tool name
                      (defaults to "__")
        """
        self.registry = registry
        self.server_names = registry.list_servers()

        # Validate and initialize tool_filter
        if tool_filter is not None:
            # Validate tool_filter contains only lists or None
            for server, tools in tool_filter.items():
                if tools is not None and not isinstance(tools, list):
                    raise ValueError(
                        f"Invalid tool_filter for server '{server}': "
                        f"value must be a list or None, got {type(tools).__name__}"
                    )

                # Validate consistent filter type (all positive or all negative)
                if (
                    tools is not None
                    and any(t.startswith("-") for t in tools)
                    and any(not t.startswith("-") for t in tools)
                ):
                    raise ValueError(
                        f"Mixed filter types for server '{server}'. "
                        f"Use either all positive filters or all negative filters."
                    )

            self.tool_filter = tool_filter
        else:
            self.tool_filter = {}

        self._namespaced_tool_map: dict[str, NamespacedTool] = {}
        self.separator = separator

        # Connection strategy
        self.transient = os.getenv("LLMPROC_MCP_TRANSIENT", "false").lower() in {
            "1",
            "true",
            "yes",
        }

        # Persistent client store (only used when transient is False)
        self._client_cms: dict[str, _PersistentClient] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

        def _close_all() -> None:  # pragma: no cover – teardown helper
            """Helper to close lingering clients if needed."""
            if self.transient or not self._client_cms:
                return
            loop = self._loop
            if loop is None or loop.is_closed():
                logger.debug("Skipping MCP client cleanup: no usable event loop")
                return
            try:
                if loop.is_running():
                    # Add timeout parameter to close_clients
                    fut = asyncio.run_coroutine_threadsafe(self.close_clients(client_timeout=0.5), loop)
                    # Set a shorter timeout for the result to ensure exit doesn't hang
                    fut.result(timeout=2)
                else:
                    # Wrap close_clients in wait_for for additional safety
                    coro = asyncio.wait_for(self.close_clients(client_timeout=0.5), timeout=2)
                    loop.run_until_complete(coro)
            except TimeoutError:
                logger.warning("Timeout while closing MCP clients during exit")
            except Exception as exc:  # noqa: BLE001 – best-effort teardown
                logger.warning("Failed to close MCP clients: %s", exc)

        atexit.register(_close_all)

    async def _get_or_create_client(self, server_name: str) -> ClientSession:
        """Return a ClientSession for *server_name*.

        In persistent mode this reuses an existing connection. In transient mode
        callers should use :py:meth:`ServerRegistry.get_client` directly and
        manage the context themselves.
        """
        if self.transient:
            raise RuntimeError("Persistent MCP connections disabled; use registry.get_client")

        if server_name in self._client_cms:
            return await self._client_cms[server_name].start()

        cm = self.registry.get_client(server_name)
        client = _PersistentClient(cm)
        self._client_cms[server_name] = client
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return await client.start()

    async def close_clients(self, client_timeout: float = 1.0) -> None:  # pragma: no cover - retained for API
        """Close all persistent MCP client sessions.

        Args:
            client_timeout: Maximum time in seconds to wait for each client to close (default: 1.0)
        """
        if self.transient:
            return

        for server, client in list(self._client_cms.items()):
            try:
                # Use asyncio.wait_for to add timeout for each client
                await asyncio.wait_for(client.close(), timeout=client_timeout)
            except TimeoutError:
                logger.warning(f"Timeout closing MCP server '{server}' after {client_timeout} seconds")
            except Exception as exc:  # noqa: BLE001 – best-effort during teardown
                logger.warning(
                    "Force-closing MCP server '%s' due to shutdown error: %s",
                    server,
                    exc,
                )
            finally:
                self._client_cms.pop(server, None)

    async def load_servers(self, specific_servers: list[str] | None = None):
        """
        Discover and namespace tools from sub-servers.

        This method connects to the specified servers (or all servers if none specified),
        retrieves their available tools, and creates namespaced versions of those tools
        that can be called through the aggregator.

        Args:
            specific_servers: Optional list of specific server names to load.
                             If None, loads all servers in the registry.

        Returns:
            None

        Note:
            This method is called automatically when listing tools or calling a tool,
            so you typically don't need to call it directly unless you want to
            preload the tools.
        """
        # Determine which servers to load
        servers_to_load = specific_servers or self.server_names

        # Only log when loading multiple servers
        if len(servers_to_load) > 1:
            logger.debug(f"Loading tools from servers: {servers_to_load}")
        elif len(servers_to_load) == 1:
            logger.debug(f"Loading tools from server: {servers_to_load[0]}")
        else:
            logger.debug("No servers to load")
            return

        # Only clear tools for servers we're loading
        if specific_servers:
            # Selectively remove tools from specific servers
            for name, tool in list(self._namespaced_tool_map.items()):
                if tool.server_name in specific_servers:
                    del self._namespaced_tool_map[name]
        else:
            # Clear all tools if loading everything
            self._namespaced_tool_map.clear()

        async def load_server_tools(server_name: str):
            """Load tool metadata from *server_name* with a 10-second timeout."""
            try:
                async with asyncio.timeout(10):
                    if self.transient:
                        async with self.registry.get_client(server_name) as client:
                            result: ListToolsResult = await client.list_tools()
                    else:
                        client = await self._get_or_create_client(server_name)
                        result: ListToolsResult = await client.list_tools()
                    tools = result.tools or []
                    logger.debug("Loaded %s tools from %s", len(tools), server_name)
                    return server_name, tools
            except Exception as e:  # noqa: BLE001 – robust against any failure
                logger.error("Error loading tools from %s: %s", server_name, e)
                return server_name, []

        # Load tools from all servers concurrently
        results = await gather(*(load_server_tools(name) for name in servers_to_load))

        # Helper function to check if a tool should be included based on the filter settings
        def should_include_tool(server_name: str, tool_name: str) -> bool:
            """Determine if a tool should be included based on the filter settings."""
            # No filter defined for this server - include all tools
            if server_name not in self.tool_filter:
                return True

            # Filter is None - include all tools from this server
            if self.tool_filter[server_name] is None:
                return True

            # Get the tool list for this server
            tool_list = self.tool_filter[server_name]

            # Empty list means include nothing
            if not tool_list:
                return False

            # Determine if we're using negative filtering
            is_negative_filter = tool_list[0].startswith("-")

            if is_negative_filter:
                # Negative filtering: include tool if NOT in the exclusion list
                return not any(t[1:] == tool_name for t in tool_list)
            else:
                # Positive filtering: include tool if in the inclusion list
                return tool_name in tool_list

        # Process and namespace the tools with filtering
        for server_name, tools in results:
            for tool in tools:
                original_name = tool.name

                # Skip this tool if it should be filtered out
                if not should_include_tool(server_name, original_name):
                    continue

                namespaced_name = f"{server_name}{self.separator}{original_name}"
                # Create a copy of the tool with the namespaced name
                namespaced_tool = tool.model_copy(update={"name": namespaced_name})
                # Add server name to the description for clarity
                namespaced_tool.description = f"[{server_name}] {tool.description or ''}"
                # Store the tool in our map
                self._namespaced_tool_map[namespaced_name] = NamespacedTool(
                    tool=namespaced_tool,
                    server_name=server_name,
                    namespaced_tool_name=namespaced_name,
                    original_name=original_name,
                )

    async def list_tools(self, return_server_mapping: bool = False) -> ListToolsResult | dict[str, list[Tool]]:
        """
        List all available tools from all sub-servers.

        This method retrieves all tools from all configured servers, applying
        namespacing to make tool names unique. It can return tools in two formats:
        either as a standard ListToolsResult with namespaced tools, or as a dictionary
        mapping server names to their original (non-namespaced) tools.

        Args:
            return_server_mapping: If True, returns a dict mapping server names to their tools without namespacing.
                                  If False, returns a ListToolsResult with all namespaced tools.

        Returns:
            Union[ListToolsResult, dict[str, list[Tool]]]: Either a ListToolsResult with namespaced tools,
            or a dictionary mapping server names to lists of their non-namespaced tools.

        Example:
            ```python
            # Get a standard ListToolsResult with namespaced tools
            tools_result = await aggregator.list_tools()

            # Get a dictionary mapping server names to their tools
            server_tools = await aggregator.list_tools(return_server_mapping=True)
            memory_tools = server_tools.get("memory", [])
            ```
        """
        # First ensure all tools are loaded
        await self.load_servers()

        if return_server_mapping:
            # Build a map of server name to list of tools
            server_tools: dict[str, list[Tool]] = {}
            for nt in self._namespaced_tool_map.values():
                server_name = nt.server_name
                # Create a copy of the tool with its original name
                original_tool = nt.tool.model_copy(update={"name": nt.original_name})

                if server_name not in server_tools:
                    server_tools[server_name] = []
                server_tools[server_name].append(original_tool)
            return server_tools

        # Default behavior: return ListToolsResult with namespaced tools
        tools = [nt.tool for nt in self._namespaced_tool_map.values()]
        result_dict = {"tools": []}

        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "inputSchema"):
                tool_dict = {"name": tool.name, "inputSchema": tool.inputSchema}
                if hasattr(tool, "description") and tool.description:
                    tool_dict["description"] = tool.description
                result_dict["tools"].append(tool_dict)

        return ListToolsResult(**result_dict)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict | None = None,
        server_name: str | None = None,
    ) -> CallToolResult:
        """
        Call a tool by its namespaced name.

        Args:
            tool_name: The tool name to call. This can be a namespaced name (server__tool)
                or just the tool name if server_name is provided separately.
            arguments: Optional dictionary of arguments to pass to the tool
            server_name: Optional server name if not included in tool_name

        Returns:
            CallToolResult with the result of the tool call
        """
        # Determine server and tool names from parameters or the namespaced string
        if server_name:
            actual_server = server_name
            actual_tool = tool_name
        else:
            if self.separator not in tool_name:
                err_msg = f"Tool name '{tool_name}' must be namespaced as 'server{self.separator}tool'"
                return CallToolResult(
                    isError=True,
                    message=err_msg,
                    content=[TextContent(type="text", text=err_msg)],
                )
            actual_server, actual_tool = tool_name.split(self.separator, 1)

        # Only load tools from the specific server we need
        # This is more efficient than loading all servers
        await self.load_servers(specific_servers=[actual_server])

        if actual_server not in self.registry.list_servers():
            err_msg = f"Server '{actual_server}' not found in registry"
            return CallToolResult(
                isError=True,
                message=err_msg,
                content=[TextContent(type="text", text=err_msg)],
            )

        # Helper function to create error result
        def error_result(message: str) -> CallToolResult:
            return CallToolResult(
                isError=True,
                message=message,
                content=[TextContent(type="text", text=message)],
            )

        # Check if the tool exists in our namespaced_tool_map
        namespaced_tool_name = f"{actual_server}{self.separator}{actual_tool}"
        if namespaced_tool_name not in self._namespaced_tool_map:
            if actual_server in self.tool_filter and self.tool_filter[actual_server] is not None:
                # The tool might be filtered out
                if actual_tool not in self.tool_filter[actual_server]:
                    return error_result(f"Tool '{actual_tool}' not found or filtered out from server '{actual_server}'")

        # Process the result from the server
        def process_result(result) -> CallToolResult:
            # If the call returns an error result, propagate it.
            if getattr(result, "isError", False):
                # Extract detailed error information
                error_message = getattr(result, "message", "")
                error_content = getattr(result, "content", [])

                # Build comprehensive error message for logging
                detailed_msg = f"MCP server '{actual_server}' returned error for tool '{actual_tool}'"
                if error_message:
                    detailed_msg += f": {error_message}"

                # Add content information if available
                if error_content:
                    content_texts = []
                    for item in error_content:
                        if hasattr(item, "text"):
                            content_texts.append(item.text)
                        elif isinstance(item, dict) and "text" in item:
                            content_texts.append(item["text"])
                    if content_texts:
                        detailed_msg += f" | Content: {' | '.join(content_texts)}"

                # Add result object structure for debugging if no useful message found
                if not error_message and not error_content:
                    available_attrs = [attr for attr in dir(result) if not attr.startswith("_")]
                    detailed_msg += f" | Available attributes: {available_attrs}"

                logger.error(detailed_msg)

                # Return simplified error message for tool result (keeping it clean for now)
                simple_msg = f"Server '{actual_server}' returned error"
                if error_message:
                    simple_msg += f": {error_message}"

                return error_result(simple_msg)

            # Process returned content into a proper list of content objects.
            content = []
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
            if not content:
                content = [TextContent(type="text", text="Tool execution completed.")]
            return CallToolResult(isError=False, message="", content=content)

        try:
            # Get tool call timeout from environment variable or use default
            tool_call_timeout = float(os.environ.get("LLMPROC_TOOL_CALL_TIMEOUT", MCP_DEFAULT_TOOL_CALL_TIMEOUT))

            if self.transient:
                async with self.registry.get_client(actual_server) as client:
                    async with asyncio.timeout(tool_call_timeout):
                        result = await client.call_tool(actual_tool, arguments)
                        return process_result(result)
            else:
                client = await self._get_or_create_client(actual_server)
                async with asyncio.timeout(tool_call_timeout):
                    result = await client.call_tool(actual_tool, arguments)
                    return process_result(result)

        except TimeoutError:
            # Get server info for more detailed error message
            server_info = f"Server type: {self.registry.registry[actual_server].type}"
            if self.registry.registry[actual_server].type == "sse":
                server_info += f", URL: {self.registry.registry[actual_server].url}"
            elif self.registry.registry[actual_server].type == "stdio":
                server_info += f", Command: {self.registry.registry[actual_server].command}"

            tool_call_timeout = float(os.environ.get("LLMPROC_TOOL_CALL_TIMEOUT", MCP_DEFAULT_TOOL_CALL_TIMEOUT))

            err_msg = MCP_ERROR_TOOL_CALL_TIMEOUT.format(
                tool=actual_tool, server=actual_server, timeout=tool_call_timeout
            )
            err_msg += f" {server_info}"

            logger.error(err_msg)
            return error_result(err_msg)
        except Exception as e:  # noqa: BLE001 – propagate as Tool error
            err_msg = f"Error in call_tool for '{tool_name}': {e}"
            logger.error(err_msg)
            return error_result(err_msg)
