# Tool Aliases

Tool aliases allow you to provide more LLM-friendly names for tools. This is particularly important for improving the likelihood that the LLM will use the tools effectively. Aliases are especially helpful for MCP tools, which often have lengthy namespaced identifiers that are hard for the model to understand and use.

## Configuration

You can define tool aliases directly on each tool entry in your TOML configuration:

```yaml
tools:
  builtin:
    - name: read_file
      alias: read
    - name: calculator
      alias: calc
    - name: list_dir
      alias: dir
```

For MCP tools, you can alias the namespaced tool name:

```yaml
mcp:
  config_path: "config/mcp_servers.json"

tools:
  mcp:
    everything:
      - name: add
        alias: add  # MCP tool alias
```

## Using Aliases in System Prompts

When using tool aliases, it's helpful to mention them in your system prompt:

```yaml
prompt:
  system_prompt: |
    You are a helpful assistant with access to tools.

    The following tools are available through easy-to-use aliases:
    - 'read': Reads a file from the local filesystem
    - 'calc': Performs mathematical calculations
    - 'dir': Lists files in a directory

    Please use these simple names when invoking tools.
```

## API Usage

You can also define tool aliases programmatically using `ToolConfig`:

```python
from llmproc.program import LLMProgram

program = LLMProgram(
    model_name="claude-3-5-haiku-20241022",
    provider="anthropic",
    system_prompt="You are a helpful assistant with access to tools.",
    tools={"builtin": [
        {"name": "calculator", "alias": "calc"},
        {"name": "read_file", "alias": "read"},
    ]}
)
```

## How Aliases Work

Each tool sets its alias directly on ``ToolMeta.name``. The registry registers
the tool under this name and exposes it unchanged. This means there is no
separate alias map to manage. Aliases are automatically applied when:

1. Tool schemas are sent to the LLM API (the LLM sees only the alias names)
2. The LLM calls a tool by its alias (the system resolves it to the actual tool name)

## Important Notes

- You specify aliases in the `alias` field of ``ToolConfig`` which overrides the
  tool's ``meta.name`` during registration
- Aliases simply change the name that appears in the schema; tool calls and errors are handled transparently
- Aliases can be used for both built-in tools and MCP tools


---
[← Back to Documentation Index](index.md)
