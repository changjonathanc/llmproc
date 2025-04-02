# CLAUDE.md - Session and Repository Summary

## Repository Structure
- `src/llmproc/`: Main package directory containing LLMProcess implementation
- `examples/`: TOML configuration examples for different providers and features
- `prompts/`: Contains system prompt templates for LLM configuration
- `tests/`: Test files with ~35% coverage of codebase
- `worktrees/`: Development branches as worktrees

## Session Procedures
- Start of Session: Read README.md to get familiar with the codebase
- End of Session: Summarize the changes made during the session

## Key Commands
- Install: `uv pip install -e ".[dev,all]"`
- Start CLI: `llmproc-demo ./examples/claude_code.toml`
- Try program linking: `llmproc-demo ./examples/program_linking/main.toml`
- Manage dependencies: `uv add <package>`, `uv add --dev <package>`, `uv remove <package>`
- Create worktree: `git worktree add worktrees/feature-name feature/feature-name`

## Testing Procedures
- Run standard tests: `pytest`
- Run verbose tests: `pytest -v`
- Run specific test file: `pytest tests/test_file.py`
- Run specific test class or function: `pytest tests/test_file.py::TestClass::test_function`
- Run tests with API access (requires API keys): `pytest -m llm_api`
- Test example programs with API: `pytest tests/test_example_programs.py -m llm_api`
- Test program linking: `pytest tests/test_program_linking_robust.py`
- Test program linking with API access: `pytest tests/test_program_linking_api.py -m llm_api`
- Test MCP tool integration: `pytest tests/test_mcp_tools_api.py -m llm_api`
- Debug output: Configure Python logging for `llmproc` package

### Test Coverage
- Unit tests: Cover core functionality without API calls (~35% code coverage)
- Provider tests: Mock API responses to test provider integrations
- MCP tests: Verify Model Context Protocol tool implementation
- Program Linking tests: Verify LLM-to-LLM communication features
- Example configurations tests: Verify all example TOML files work with actual APIs
- CLI tests: Verify command-line interface functionality with actual APIs
- API Integration tests: Verify end-to-end functionality with real API calls
- All tests are marked appropriately to skip API tests by default

## Code Style Guidelines
- Uses absolute imports (`from llmproc import X`)
- Type hints on all functions and methods
- Google-style docstrings
- Max line length of 200 characters (configured in Black/ruff)
- PEP8 compliant with Ruff enforcement

## LLMProcess Features
- Configurable via TOML files with validation
- Supports system prompts from strings or files
- Maintains conversation state
- Parameters configurable via TOML
- File preloading for context via system prompt (using [preload] section in TOML or preload_files() method)
- Custom display names for models in CLI interfaces
- Supports OpenAI, Anthropic, and Vertex AI models
- MCP (Model Context Protocol) support for tool usage
- Program linking for LLM-to-LLM communication via spawn tool
- File descriptor system for handling large inputs/outputs
- Token-efficient tool use (Claude 3.7+)
- Support for reasoning models (GPT-4-o-0531 and Claude 3.7)
- Command-line interface for interactive chat sessions
- Comprehensive error handling and diagnostics
- Well-documented API (see Core API Architecture section)

## TOML Configuration Structure
- `[model]`: Model name, provider, display name
- `[parameters]`: API parameters like temperature, max_tokens
- `[prompt]`: System prompt configuration
- `[preload]`: Files to preload into context
- `[tools]`: Built-in tool configuration
- `[mcp]`: MCP server configuration
- `[mcp.tools]`: External MCP tool settings
- `[file_descriptor]`: File descriptor system settings
- `[linked_programs]`: Program linking configuration
- `[env_info]`: Environment info to include in context

## Special Features

### File Descriptor System
- Handles large inputs/outputs by creating file-like references
- Configured via `[file_descriptor]` section in TOML
- Pages large content automatically
- Supports direct file creation with `fd_to_file` tool
- Reference system for linking to specific content

### Program Linking
- Connects multiple LLMProcess instances for collaboration
- Each program can have different models, configurations, etc.
- Communication via `spawn` tool
- Configured via `[linked_programs]` section in TOML

### MCP Tool Support
- Connect to external tool servers via Model Context Protocol
- Standardized interface for third-party tools
- Configured via `[mcp.tools]` section in TOML
- Supports multiple servers with different tool sets

## Core API Architecture

### LLMProgram
- **Purpose**: Configuration and compilation of LLM interactions
- **Key Methods**:
  - `from_toml(toml_path)`: Create program from TOML configuration
  - `compile()`: Validate and prepare program for execution
  - `start()`: Create fully initialized LLMProcess from this program
  - `set_enabled_tools(tool_names)`: Configure which tools are available
  - `enable_token_efficient_tools()`: Enable token-efficient tool use for Claude 3.7
  - `get_enriched_system_prompt()`: Generate full system prompt with context
  - `link_program(name, program)`: Connect to another program for interaction

### LLMProcess
- **Purpose**: Execution engine for LLM interactions
- **Key Methods**:
  - `run(user_input, max_iterations=10, callbacks=None)`: Process user input with tool support
  - `call_tool(tool_name, args)`: Invoke a tool with arguments
  - `get_state()`: Retrieve current conversation state
  - `reset_state()`: Clear conversation history
  - `preload_files(file_paths)`: Add file content to context
  - `count_tokens()`: Calculate token usage in conversation
  - `get_last_message()`: Get most recent model response
  - `fork_process()`: Create a copy with the same state

### Tool System Architecture
- **ToolManager**: Central point for managing all tool operations
  - `tool_manager.register_system_tools(process)`: Registers built-in tools like calculator, read_file
  - `tool_manager.process_function_tools()`: Processes tools defined with function decorators
  - `tool_manager.add_function_tool(func)`: Adds individual function-based tools
  - `tool_manager.call_tool(name, args)`: Calls a tool by name with arguments
  - `tool_manager.get_tool_schemas()`: Gets tool definitions for LLM API
  - `tool_manager.set_enabled_tools(tool_names)`: Controls which tools are available
- **ToolRegistry**: Internal store for tool definitions and handlers
  - `register_tool(name, handler, definition)`: Registers individual tools
  - `call_tool(name, args)`: Executes a tool by name
  - `get_definitions()`: Returns all registered tool definitions
- **MCP Tools Integration**: Connect to external tool servers
  - `mcp.initialize_mcp_tools(process, tool_registry, config_path, tools_config)`: Initializes MCP tools
  - MCP tools use namespaced names with server__toolname format
  - TOML configuration: `[mcp.tools]` section specifies which MCP tools to enable