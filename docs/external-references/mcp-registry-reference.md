# MCP Registry Reference

The Model Context Protocol (MCP) provides a way for LLMs to use external tools. The MCP Registry is a key component that manages tool registration and execution.

## MCPAggregator Class

The `MCPAggregator` class aggregates multiple MCP servers and provides a unified interface for tool calls.

```python
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