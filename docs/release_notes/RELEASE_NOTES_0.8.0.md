# Release Notes - v0.8.0

## 🎉 Major Features

### New Synchronous API
- Added `program.start_sync()` method for synchronous process creation
- Returns `SyncLLMProcess` with blocking calls instead of async/await
- See [Persistent Event Loop documentation](docs/persistent-event-loop.md) for examples

### YAML and Dictionary Configuration Support
- Full YAML configuration support alongside TOML format
- New `from_dict()` method for creating programs from Python dictionaries
- Dynamic configuration generation without files
- See [YAML Configuration Schema](docs/yaml_config_schema.md) for details

### Enhanced MCP (Model Context Protocol) Support
- **New `MCPServerTools` class** replaces previous MCP tool registration
- Embedded MCP server configurations directly in TOML/YAML files
- Tool description override support for customizing tool descriptions
- No longer requires separate `mcp_servers.json` files
- See [MCP Feature documentation](docs/mcp-feature.md) for configuration examples

## 🚀 New Features

### New CLI
- **`llmproc`** - New command for single prompt execution (non-interactive)
- **`llmproc-demo`** - Interactive chat interface (previously the only CLI)
- Better separation: use `llmproc` for scripts/automation, `llmproc-demo` for interactive sessions

### Instance Methods as Tools
- Register instance methods directly as tools for stateful implementations
- See [Function-Based Tools documentation](docs/function-based-tools.md#instance-methods-as-tools)

### API Retry Configuration
- Configurable retry logic for Anthropic API calls via environment variables
- Automatic retry with exponential backoff
- See [Environment Variables documentation](docs/environment-variables.md#retry-configuration)

### Spawn Tool Self-Spawning
- Spawn tool now supports spawning the current program without linked programs
- Leave `program_name` empty to create independent instances of the same program
- Useful for parallel task execution and exploration

### Enhanced Callbacks System
- Monitor tool registration, execution, and API events
- Supports `TOOL_START`, `TOOL_END`, `RESPONSE`, `API_REQUEST`, `API_RESPONSE`, `TURN_START`, `TURN_END`, and `STDERR_WRITE` events
- **New support for async callback methods** alongside synchronous functions
- Transparently handles both async and sync callbacks in the same implementation
- Callback classes can freely mix sync and async methods
- See [Callbacks documentation](docs/callbacks.md) for usage examples

### New Tool: Write to Standard Error
- New built-in `write_stderr` tool allows LLMProcess to have stderr output
- Inspired by Unix processes with distinct stdin, stdout, and stderr while LLM processes previously only had input and output
- Provides a logging channel so messages can be reviewed later without exposing the full conversation
- With the description override config option you can rename and repurpose the tool, e.g. piping the log to another LLM process
- Integrated with CLI callback system for proper stderr handling (see `cli/run.py` for example usage)
- Accessible via Callback `stderr_write(text)` or `LLMProcess.get_stderr_log()`
- **Experimental feature**

### Unified ToolConfig for MCP and Built-in Tools
- Now MCP and built-in tools share the same ToolConfig
- Supports alias, description overrides, param description override
- Works in both YAML and TOML formats

```yaml
tools:
  builtin:
    - name: "write_stderr"
      alias: "write_log"
      description: "append a new message to the work log"
      param_descriptions:
        message: "a message to be logged"

```

### Tool Configuration Naming
- New `builtin` field name for built-in tools (more semantic alongside `mcp`)
- Previous `enabled` field still works for backward compatibility
- Example: `tools.builtin` instead of `tools.enabled`

### Better Error Handling
- Graceful handling of incorrect tool names - returns error results instead of crashing
- Validation for duplicate tool names and aliases
- Improved configuration validation with Pydantic models
- Clearer error messages for missing runtime context

## 📝 Breaking Changes and Migration Guide

### MCP Tool Registration
The old MCP tool registration API has been replaced with the new `MCPServerTools` class:
```python
# New way
from llmproc import MCPServerTools
register_tools([MCPServerTools(server="weather", tools=["get_forecast"])])
```

### Tool Aliases and Description Overrides
Tool aliases and description overrides are now supported in both YAML and TOML formats:

```yaml
tools:
  builtin:
    - name: "read_file"
      alias: "read"
      description: "Read any file from the filesystem"
```

### Configuration Files
YAML format is now supported for all configuration needs. Use `.yaml` extension with the same structure as TOML files.

## 🐛 Bug Fixes
- Fixed MCP cleanup handling during shutdown
- Resolved circular dependency issues
- Improved async/sync interface reliability
- Fixed configuration validation edge cases

## 📚 Examples

### Complete Usage Example
Here's a practical example demonstrating multiple v0.8.0 features:

```bash
# List available builtin tools using YAML configuration with MCP tools
llmproc ./examples/min_claude_code_read_only.yaml -p 'give me a list of builtin tools in llmproc'
```

This example showcases:
- YAML configuration format (new in v0.8.0)
- MCP tool integration with embedded server configuration (new in v0.8.0)
- Tool description override for the Read tool (new in v0.8.0)
- Non-interactive CLI usage with `llmproc` cli tool (new in v0.8.0)
- Tool aliases for simpler tool names

## 📦 Compatibility Note
While v0.8.0 maintains backward compatibility for most features, the MCP tool registration has breaking changes. Please review the Breaking Changes section above for migration details.

---

For detailed API documentation and more examples, visit the [documentation](https://github.com/cccntu/llmproc/tree/main/docs).
