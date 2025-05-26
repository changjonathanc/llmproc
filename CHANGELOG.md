# Changelog

## [0.8.0] - 2025-05-25

See [detailed release notes](docs/release_notes/RELEASE_NOTES_0.8.0.md) for complete information and migration guide.

### Added
- **Synchronous API Support**: New `program.start_sync()` method returns `SyncLLMProcess` for blocking operations
  - All async methods now have synchronous counterparts
  - Automatic event loop management for synchronous codebases
  
- **Dictionary & YAML Configuration**: Added support for Python dictionaries and YAML format
  - `LLMProgram.from_dict(config_dict)` for dynamic configuration
  - YAML configuration alternative to TOML
  
- **MCP Enhancements**:
  - New `MCPServerTools` class for tool registration
  - Embedded MCP server configurations directly in TOML/YAML
  - Tool description override support
  
- **Dual CLI**:
  - `llmproc` for single prompt execution (non-interactive)
  - `llmproc-demo` for interactive chat (previously the only CLI)
  
- **Instance Methods as Tools**: Register instance methods directly as tools
  
- **API Retry Configuration**: Configurable retry logic via environment variables
  
- **Spawn Tool Self-Spawning**: Create independent instances of the same program
  
- **Enhanced Callbacks**: Support for async callback methods and new event types
  
- **Write to Standard Error**: New built-in `write_stderr` tool
  
- **Unified ToolConfig**: Shared configuration for MCP and built-in tools

### Changed
- **Tool Configuration Naming**: New `builtin` field for tools (alongside existing `enabled`)
- **MCP Tool Registration**: Now uses the `MCPServerTools` class

### Fixed
- MCP cleanup handling during shutdown
- Improved async/sync interface reliability
- Configuration validation edge cases
- Better error handling for incorrect tool names

## [0.7.0] - 2025-05-01

Initial version tracked in this changelog.
