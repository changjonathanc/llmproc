# CodeMCP Usage Review

This document compiles all references to "codemcp" in the codebase for review.

## Overview

CodeMCP appears to be a tool integration using the Model Context Protocol (MCP) framework. It provides file system access capabilities to LLM processes, with tools such as:

- `ReadFile`/`read_file` - Reading files from the filesystem
- `ls` - Listing directory contents
- `grep` - Searching file contents
- `write_file` - Writing to files (mentioned but not always enabled)
- `edit_file` - Editing existing files (mentioned but not always enabled)
- `init_project_tool` - Initializing projects (mentioned but not always enabled)

## Configuration References

### MCP Servers Configuration

In `/Users/ttj/github/agent-process/llmproc/tests/test_mcp_features.py`, CodeMCP is configured as:

```json
"codemcp": {
    "type": "stdio",
    "command": "/bin/zsh",
    "args": [
        "-c",
        "uvx --from git+https://github.com/cccntu/codemcp@main codemcp "
    ]
}
```

The CodeMCP tool is loaded from a GitHub repository using `uvx` with the URL `git+https://github.com/cccntu/codemcp@main`.

## TOML Configuration Examples

### Claude Code Configuration

In `/Users/ttj/github/agent-process/llmproc/examples/claude-code/claude-code.toml`:

```toml
[mcp]
config_path = "../../config/mcp_servers.json"

[mcp.tools]
# this is the full list of tools
#codemcp = ["read_file", "write_file", "edit_file", "ls", "grep", "init_project_tool"]
# we only allow read only access to the file system for demo purposes
codemcp = ["read_file", "ls", "grep"]
```

The Claude Code example only enables read-only tools (`read_file`, `ls`, `grep`) for demonstration purposes, while commenting out write operations.

### Dispatch Agent Configuration

In `/Users/ttj/github/agent-process/llmproc/examples/claude-code/dispatch-agent.toml`:

```toml
[mcp]
config_path = "../../config/mcp_servers.json"

[mcp.tools]
# this is the full list of tools
#codemcp = ["read_file", "write_file", "edit_file", "ls", "grep", "init_project_tool"]
# we only allow read only access to the file system for demo purposes
codemcp = ["read_file", "ls", "grep"]
```

The Dispatch Agent has identical configuration to Claude Code, with the same read-only tools enabled.

### Fork Tool Example

In `/Users/ttj/github/agent-process/llmproc/examples/features/fork.toml`:

```toml
[mcp]
config_path = "../../config/mcp_servers.json"

[mcp.tools]
# Only configure servers that are available in your config/mcp_servers.json
codemcp = ["read_file"]
```

The Fork tool example only enables the `read_file` tool from CodeMCP.

## Documentation References

In `/Users/ttj/github/agent-process/llmproc/docs/mcp-feature.md`, CodeMCP is referenced as an example MCP tool:

```toml
[mcp.tools]
github = ["search_repositories", "get_file_contents"]
codemcp = ["ReadFile"]
```

The MCP documentation also shows CodeMCP configuration in the server configuration example:

```json
"codemcp": {
  "type": "stdio",
  "command": "/bin/zsh",
  "args": [
    "-c",
    "uvx --from git+https://github.com/cccntu/codemcp@main codemcp "
  ]
}
```

## Test References

In `/Users/ttj/github/agent-process/llmproc/tests/test_mcp_features.py`, CodeMCP is used in tests:

```python
mock_tool3 = MagicMock()
mock_tool3.name = "codemcp.ReadFile"
mock_tool3.description = "Read a file from the filesystem"
mock_tool3.inputSchema = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
}
```

Tests verify that CodeMCP tools are properly configured:

```python
assert process.mcp_tools == {
    "github": ["search_repositories"],
    "codemcp": ["ReadFile"],
}
```

The testing documentation in `/Users/ttj/github/agent-process/llmproc/docs/api_testing.md` mentions testing CodeMCP tool functionality:

```
`test_mcp_tool_functionality`: Tests the Model Context Protocol functionality:
- Verifies tool registration
- Tests tool execution with the codemcp tool
- Confirms the model can use tools to access file content
```

## Summary

CodeMCP is used as a file system access tool within the MCP framework, primarily providing read-only capabilities (`read_file`, `ls`, `grep`) in examples, while full capabilities include file writing and editing. It's loaded from the GitHub repository `https://github.com/cccntu/codemcp`. The codebase includes tests to verify proper configuration and functionality of CodeMCP tools.