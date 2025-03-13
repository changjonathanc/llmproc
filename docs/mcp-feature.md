# Model Context Protocol (MCP) Feature

The Model Context Protocol (MCP) feature integrates tool usage capabilities with LLM models, allowing LLMs to use external tools to perform actions like searching the web, accessing GitHub repositories, reading files, and more.

## Overview

MCP provides a standardized way for LLM agents to interact with external tools. This feature enables the LLMProcess to:

1. Connect to MCP servers through a central registry
2. Register specific tools with the LLM model
3. Handle tool calls during model generation
4. Process tool results and continue the conversation

## Configuration

MCP is configured through the TOML configuration file with a dedicated `[mcp]` section:

```toml
[mcp]
config_path = "config/mcp_servers.json"

[mcp.tools]
github = ["search_repositories", "get_file_contents"]
codemcp = ["ReadFile"]
```

### Configuration Options

- `config_path`: Path to the MCP servers configuration JSON file
- `[mcp.tools]`: Dictionary of server names mapped to tools to enable
  - Server name = List of specific tools to import OR "all" to import all tools from that server

### MCP Servers Configuration

The MCP servers configuration file specifies the servers to connect to:

```json
{
  "mcpServers": {
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "codemcp": {
      "type": "stdio",
      "command": "/bin/zsh",
      "args": [
        "-c",
        "uvx --from git+https://github.com/cccntu/codemcp@main codemcp "
      ]
    }
  }
}
```

## Tool Filtering

Users must explicitly specify which tools they want to import from each server. Even if the MCP config file includes many servers, only the explicitly specified tools will be added to the LLM process. This is enforced for security and performance reasons.

You can specify tools in two ways:

1. List specific tools: `github = ["search_repositories", "get_file_contents"]`
2. Import all tools from a server: `github = "all"`

## Provider Support

Currently, MCP functionality is only supported with the Anthropic provider. Support for OpenAI providers will be added in a future update.

## Example Usage

### TOML Configuration

```toml
[model]
name = "claude-3-haiku-20240307"
provider = "anthropic"
display_name = "Claude MCP Assistant"

[parameters]
temperature = 0.7
max_tokens = 300

[prompt]
system_prompt = "You are a helpful assistant with access to tools. Use tools whenever appropriate to answer user queries accurately."

[mcp]
config_path = "config/mcp_servers.json"

[mcp.tools]
github = ["search_repositories", "get_file_contents"]
codemcp = ["ReadFile"]
```

### Python Code

```python
from llmproc import LLMProcess

# Initialize from TOML configuration
llm = LLMProcess.from_toml("examples/mcp.toml")

# Use the LLM with tools
response = llm.run("Please search for popular Python repositories on GitHub.")
print(response)
```

## Implementation Details

The implementation includes:

1. Extending LLMProcess to handle MCP configuration
2. Tool registration and filtering based on user configuration
3. Handling tool calls in Anthropic API responses
4. Processing tool results and continuing the conversation

Tools are initialized asynchronously when the LLMProcess is created, allowing for dynamic tool discovery and initialization.

## Future Enhancements

- Support for OpenAI's function calling API
- Enhanced tool call formatting and debugging
- Custom tool implementations
- Persistent tool state across conversation turns