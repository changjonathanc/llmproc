# MCP Registry Reference

The Model Context Protocol (MCP) provides a way for LLMs to use external tools. The MCP Registry is a key component that manages tool registration and execution.

## MCPAggregator Class

The `MCPAggregator` class aggregates multiple MCP servers and provides a unified interface for tool calls.

```python
import asyncio
import json
import sys
import logging
import anyio
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
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    ListToolsResult,
    Prompt,
    ResourceTemplate,
    TextContent,
    Tool,
)
from pydantic import BaseModel

# Set up logging for debugging purposes.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MCPServerSettings(BaseModel):
    """
    Basic server configuration settings.
    """
    type: str  # "stdio" or "sse"
    command: str | None = None  # for stdio
    args: list[str] | None = None  # for stdio
    url: str | None = None  # for sse
    env: dict | None = None
    description: str | None = None

    @property
    def transport(self) -> str:
        return self.type

    @transport.setter
    def transport(self, value: str) -> None:
        self.type = value


class ServerRegistry:
    """
    Simple registry for managing server configurations.
    """
    def __init__(self, servers: dict[str, MCPServerSettings]):
        self.registry = servers

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
        servers = {
            name: MCPServerSettings(**settings)
            for name, settings in config.items()
        }
        return cls(servers)

    def save_config(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "mcpServers": {
                name: settings.model_dump(exclude_none=True)
                for name, settings in self.registry.items()
            }
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_config(cls, path: Path | str) -> "ServerRegistry":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            config = json.load(f)
        if "mcpServers" not in config:
            raise KeyError("Config file must have a 'mcpServers' section")
        servers = {
            name: MCPServerSettings(**settings)
            for name, settings in config["mcpServers"].items()
        }
        return cls(servers)

    @asynccontextmanager
    async def get_client(self, server_name: str) -> AsyncGenerator[ClientSession, None]:
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
        return list(self.registry.keys())

    def get_server_info(self, server_name: str) -> str:
        if server_name not in self.registry:
            return f"Server '{server_name}' not found"
        config = self.registry[server_name]
        desc = f" - {config.description}" if config.description else ""
        type_info = (
            f"stdio: {config.command} {' '.join(config.args or [])}"
            if config.type == "stdio"
            else f"sse: {config.url}"
        )
        return f"{server_name}: {type_info}{desc}"


class NamespacedTool(BaseModel):
    """
    A tool that is namespaced by server name.
    """
    tool: Tool
    server_name: str
    namespaced_tool_name: str
    original_name: str


class MCPAggregator:
    """
    Aggregates multiple MCP servers.

    This class can be used in two ways:
    1. As a regular object (default) - creates temporary connections for each tool call
    2. As an async context manager - maintains persistent connections during the context

    Example:
        # Method 1: Temporary connections (default behavior)
        aggregator = MCPAggregator(registry)
        result = await aggregator.call_tool("memory__get", {"key": "test"})

        # Method 2: Persistent connections with context manager
        async with MCPAggregator(registry) as aggregator:
            # All tool calls in this block will use persistent connections
            result1 = await aggregator.call_tool("memory__get", {"key": "test"})
            result2 = await aggregator.call_tool("memory__set", {"key": "test", "value": "hello"})
    """
    def __init__(self, registry: ServerRegistry, server_names: list[str] | None = None, separator: str = "__")m
        self.registry = registry
        self.server_names = server_names or registry.list_servers()
        self._namespaced_tool_map: dict[str, NamespacedTool] = {}
        self._connection_manager = None
        self._in_context_manager = False
        self.separator = separator

    async def __aenter__(self):
        """Enter the context manager - initialize persistent connections."""
        # Import here to avoid circular imports
        from mcp_registry.connection import MCPConnectionManager

        self._in_context_manager = True
        self._connection_manager = MCPConnectionManager(self.registry)
        await self._connection_manager.__aenter__()

        # Preload the tools
        await self.load_servers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager - close all persistent connections."""
        if self._connection_manager:
            await self._connection_manager.__aexit__(exc_type, exc_val, exc_tb)
            self._connection_manager = None
        self._in_context_manager = False

    async def load_servers(self, specific_servers: list[str] | None = None):
        """
        Discover and namespace tools from sub-servers.

        Args:
            specific_servers: Optional list of specific server names to load.
                              If None, loads all servers in self.server_names.
        """
        # Determine which servers to load
        servers_to_load = specific_servers or self.server_names

        # Only log when loading multiple servers
        if len(servers_to_load) > 1:
            logger.info(f"Loading tools from servers: {servers_to_load}")
        elif len(servers_to_load) == 1:
            logger.info(f"Loading tools from server: {servers_to_load[0]}")
        else:
            logger.info("No servers to load")
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
            try:
                async with asyncio.timeout(10):
                    # Use persistent connection if available, otherwise create temporary one
                    if self._in_context_manager and self._connection_manager:
                        server_conn = await self._connection_manager.get_server(server_name)
                        if server_conn and server_conn.session:
                            result: ListToolsResult = await server_conn.session.list_tools()
                            tools = result.tools or []
                            logger.info(f"Loaded {len(tools)} tools from {server_name} (persistent)")
                            return server_name, tools

                    # Fallback to temporary connection
                    async with self.registry.get_client(server_name) as client:
                        result: ListToolsResult = await client.list_tools()
                        tools = result.tools or []
                        logger.info(f"Loaded {len(tools)} tools from {server_name} (temporary)")
                        return server_name, tools
            except Exception as e:
                logger.error(f"Error loading tools from {server_name}: {e}")
                return server_name, []

        results = await gather(*(load_server_tools(name) for name in servers_to_load))
        for server_name, tools in results:
            for tool in tools:
                original_name = tool.name
                namespaced_name = f"{server_name}{self.separator}{original_name}"
                namespaced_tool = tool.model_copy(update={"name": namespaced_name})
                namespaced_tool.description = f"[{server_name}] {tool.description or ''}"
                self._namespaced_tool_map[namespaced_name] = NamespacedTool(
                    tool=namespaced_tool,
                    server_name=server_name,
                    namespaced_tool_name=namespaced_name,
                    original_name=original_name,
                )

    async def list_tools(self, return_server_mapping: bool = False) -> ListToolsResult | dict[str, list[Tool]]:
        """
        List all available tools from all sub-servers.

        Args:
            return_server_mapping: If True, returns a dict mapping server names to their tools without namespacing.
                                 If False, returns a ListToolsResult with all namespaced tools.

        Returns:
            Union[ListToolsResult, dict[str, list[Tool]]]: Either a ListToolsResult with namespaced tools,
            or a dictionary mapping server names to lists of their non-namespaced tools.
        """
        await self.load_servers()

        if return_server_mapping:
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
                tool_dict = {
                    "name": tool.name,
                    "inputSchema": tool.inputSchema
                }
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
        """
        # Determine server and tool names from parameters or the namespaced string.
        if server_name:
            actual_server = server_name
            actual_tool = tool_name
        else:
            if self.separator not in tool_name:
                err_msg = (
                    f"Tool name '{tool_name}' must be namespaced as 'server{self.separator}tool'"
                )
                return CallToolResult(
                    isError=True,
                    message=err_msg,
                    content=[TextContent(type="text", text=err_msg)],
                )
            actual_server, actual_tool = tool_name.split(self.separator, 1)

        # Only load tools from the specific server we need
        # This is more efficient than loading all servers
        await self.load_servers(specific_servers=[actual_server])

        if actual_server not in self.server_names:
            err_msg = f"Server '{actual_server}' not found or not enabled"
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

        # Process the result from either connection type
        def process_result(result) -> CallToolResult:
            # If the call returns an error result, propagate it.
            if getattr(result, "isError", False):
                err_msg = f"Server '{actual_server}' returned error: {getattr(result, 'message', '')}"
                return error_result(err_msg)

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
                    if isinstance(item, (TextContent, ImageContent, EmbeddedResource)):
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
            result = None

            # Try using persistent connection if available
            if self._in_context_manager and self._connection_manager:
                try:
                    # Get server connection from the connection manager
                    server_conn = await self._connection_manager.get_server(actual_server)
                    if server_conn and server_conn.is_ready and server_conn.session:
                        # Use persistent connection
                        async with asyncio.timeout(30):
                            result = await server_conn.session.call_tool(actual_tool, arguments)
                            logger.debug(f"Called tool using persistent connection to {actual_server}")
                            return process_result(result)
                except asyncio.TimeoutError:
                    return error_result(
                        f"Timeout calling tool '{actual_tool}' on server '{actual_server}'"
                    )
                except Exception as e:
                    logger.warning(f"Failed to use persistent connection, falling back to temporary: {e}")

            # Use temporary connection as fallback or default
            async with self.registry.get_client(actual_server) as client:
                async with asyncio.timeout(30):
                    result = await client.call_tool(actual_tool, arguments)
                    return process_result(result)

        except asyncio.TimeoutError:
            return error_result(
                f"Timeout calling tool '{actual_tool}' on server '{actual_server}'"
            )
        except Exception as e:
            err_msg = f"Error in call_tool for '{tool_name}': {e}"
            logger.error(err_msg)
            return error_result(err_msg)


async def run_registry_server(registry: ServerRegistry, server_names: list[str] | None = None):
    """
    Create and run an MCP compound server that aggregates tools from the registry.
    """
    # Create server
    server = Server("MCP Registry Server")

    # Create aggregator
    aggregator = MCPAggregator(registry, server_names)

    # List available tools
    try:
        await aggregator.load_servers()
    except Exception as e:
        logger.error(f"Error loading servers: {e}")
        return

    # Implement list_tools method
    @server.list_tools()
    async def list_tools():
        """List available tools."""
        result = await aggregator.list_tools()
        # Return the list of tools directly as expected by the MCP protocol
        return [t.model_dump() for t in result.tools]

    # Implement call_tool method
    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None = None):
        """Call a specific tool by name."""
        result = await aggregator.call_tool(tool_name=name, arguments=arguments)
        return result.content if hasattr(result, 'content') else [TextContent(type="text", text="No content")]

    # Create initialization options
    init_options = server.create_initialization_options(
        notification_options=NotificationOptions(),
        experimental_capabilities={}
    )

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        # First set up the server
        await server.run(
            read_stream,
            write_stream,
            init_options
        )

        # Then load servers
        await aggregator.load_servers()
        logger.info("MCP Registry Server ready!")

        # Wait forever or until interrupted
        try:
            while True:
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            logger.info("Registry server shutting down due to cancellation")
            # Clean up by explicitly disconnecting from all servers
            if hasattr(aggregator, "_connection_manager") and aggregator._connection_manager:
                await aggregator._connection_manager.disconnect_all()
            raise
```

### Key Methods

- `async list_tools(return_server_mapping=False)` - Lists all available tools, optionally grouped by server
- `async call_tool(tool_name, tool_args)` - Calls a tool with the given arguments
- `async __aenter__()` and `async __aexit__()` - Context manager methods for persistent connections

### Tool Naming Convention

Tools are named using the format `{server_name}__{tool_name}` where:
- `server_name` is the name of the MCP server
- `tool_name` is the name of the tool on that server

This convention ensures there are no naming conflicts when using tools from multiple servers.

## ServerRegistry Class

The `ServerRegistry` manages the connection to MCP servers and provides methods for discovering and registering tools.

### Configuration Format

```json
{
  "mcpServers": {
    "server_name": {
      "type": "stdio",
      "command": "executable",
      "args": ["arg1", "arg2"]
    }
  }
}
```

### Key Methods

- `from_config(config_path)` - Creates a registry from a configuration file
- `get_server(server_name)` - Gets a server by name

## Implementation in LLMProcess

The LLMProcess class uses these components to:
1. Initialize MCP tools based on user configuration
2. Register tools with the LLM model
3. Execute tool calls during model generation
4. Process tool results and continue the conversation