# Model Context Protocol (MCP) Feature

The Model Context Protocol (MCP) enables LLMs to interact with external tools through a standardized interface.

## Quick Start

### Basic Configuration

```yaml
# config.yaml
model:
  name: claude-3-5-sonnet-20241022
  provider: anthropic

# Embedded server configuration (NEW in v0.8.0)
mcp:
  servers:
    weather:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-weather"]
    github:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: ${GITHUB_TOKEN}

# Select specific tools
tools:
  mcp:
    weather: ["get_forecast"]  # Specific tools
    github: "all"              # All tools from server
```

### Using External Configuration

```toml
# config.toml
[mcp]
config_path = "config/mcp_servers.json"  # External server definitions

[tools.mcp]
weather = ["get_forecast"]
github = ["search_repositories", "get_file_contents"]
```

## Advanced Features

### Tool Description Override (NEW in v0.8.0)

Customize tool descriptions for better LLM understanding:

```yaml
# YAML format (recommended for complex configurations)
tools:
  mcp:
    calculator:
      add:
        description: "Add two numbers together"
      multiply:
        description: "Multiply two numbers"
      divide:
        access: read
        description: "Divide first number by second (read-only)"
```

```toml
# TOML format
[tools.mcp.calculator]
add = {description = "Add two numbers together"}
multiply = {description = "Multiply two numbers"}
divide = {access = "read", description = "Divide first number by second (read-only)"}
```

### Programmatic Configuration

```python
from llmproc import LLMProgram, MCPServerTools
from llmproc.common.access_control import AccessLevel

program = LLMProgram(...)

# Configure MCP servers
program.configure_mcp(servers={
    "calc": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-calculator"]
    }
})

# Register tools with custom access levels
program.register_tools([
    # All tools from a server
    MCPServerTools(server="calc"),

    # Specific tools with access control
    MCPServerTools(server="github", tools=["search_repos"], default_access=AccessLevel.READ),

    # Per-tool configuration
    MCPServerTools(server="files", tools=[
        {"name": "read_file", "access": "read", "description": "Read file contents"},
        {"name": "write_file", "access": "write"}
    ])
])
```

## Usage Examples

### Asynchronous Usage

```python
import asyncio
from llmproc import LLMProgram

async def main():
    program = LLMProgram.from_toml("config.toml")
    process = await program.start()

    result = await process.run("What's the weather in Tokyo?")
    print(process.get_last_message())

asyncio.run(main())
```

### Synchronous Usage

```python
from llmproc import LLMProgram

program = LLMProgram.from_toml("config.toml")
process = program.start_sync()

result = process.run("Search for Python web frameworks on GitHub")
print(process.get_last_message())
```

## Tool Naming Convention

MCP tools are namespaced with the server name:
- Format: `servername__toolname`
- Example: `github__search_repositories`, `weather__get_forecast`

This prevents naming conflicts between servers.

## Access Control

Tools can have different access levels:
- `READ`: Read-only operations
- `WRITE`: Modify data (default)
- `ADMIN`: Administrative operations

## Provider Support

Currently supported:
- ✅ Anthropic (Claude models)
- ✅ Anthropic via Vertex AI
- ❌ OpenAI (planned)

## Troubleshooting

### Common Issues

1. **Tool not found**: Ensure the tool name matches exactly (case-sensitive)
2. **Server not starting**: Check command and args in server configuration
3. **Environment variables**: Use `${VAR_NAME}` syntax for env var substitution

### Debugging

Enable debug logging to see MCP communication:
```python
import logging
logging.getLogger("llmproc.mcp").setLevel(logging.DEBUG)
```

## See Also

- [Tool Aliases](tool-aliases.md) - Simplify tool names
- [Python SDK](python-sdk.md) - Programmatic usage
- Example: `examples/mcp.toml`
