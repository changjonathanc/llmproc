# Tool Aliases

Tool aliases allow you to provide more LLM-friendly names for tools. This is particularly important for improving the likelihood that the LLM will use the tools effectively. Aliases are especially helpful for MCP tools, which often have lengthy namespaced identifiers that are hard for the model to understand and use.

## Configuration

You can define tool aliases directly on each tool entry in your TOML configuration:

```toml
[tools]
enabled = [
    {name = "read_file", alias = "read"},
    {name = "calculator", alias = "calc"},
    {name = "list_dir", alias = "dir"},
]
```

For MCP tools, you can alias the namespaced tool name:

```toml
[mcp]
config_path = "config/mcp_servers.json"

[tools.mcp]
everything = [
    {name = "add", alias = "add"}  # MCP tool alias
]
```

## Using Aliases in System Prompts

When using tool aliases, it's helpful to mention them in your system prompt:

```toml
[prompt]
system_prompt = """
You are a helpful assistant with access to tools.

The following tools are available through easy-to-use aliases:
- 'read': Reads a file from the local filesystem
- 'calc': Performs mathematical calculations
- 'dir': Lists files in a directory

Please use these simple names when invoking tools.
"""
```

## API Usage

You can also set tool aliases programmatically:

```python
from llmproc.program import LLMProgram

program = LLMProgram(
    model_name="claude-3-5-haiku-20241022",
    provider="anthropic",
    system_prompt="You are a helpful assistant with access to tools.",
    tools={"enabled": ["calculator", "read_file"]}
)

# Set aliases
program.set_tool_aliases({
    "calc": "calculator",
    "read": "read_file"
})
```

## How Aliases Work

Aliases are registered during program compilation and are automatically applied when:

1. The tool schemas are sent to the LLM API (the LLM sees only the alias names)
2. The LLM calls a tool by its alias (the system resolves it to the actual tool name)

## Important Notes

- You must enable the original tool names, not the aliases, in the `tools.builtin` (or legacy `tools.enabled`) list
- Aliases must form a one-to-one mapping (no duplicate aliases pointing to the same tool)
- When a tool is called using its alias, error messages will include both the alias and the resolved tool name
- Aliases can be used for both built-in tools and MCP tools
