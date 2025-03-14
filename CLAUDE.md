# CLAUDE.md - Session and Repository Summary

## Repository Structure
- `src/llmproc/`: Main package directory containing LLMProcess implementation
- `examples/`: TOML configuration examples (minimal.toml, complex.toml)
- `prompts/`: Contains system prompt templates for LLM configuration
- `tests/`: Test files with 90% coverage of codebase
- `worktrees/`: Development branches as worktrees

## Session Procedures
- Start of Session: Read README.md and repo-map.txt to get familiar with the codebase
- End of Session: Update repo-map.txt with any changes made during the session

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
- Test program linking: `pytest tests/test_program_linking_robust.py`
- Test program linking with API access: `pytest tests/test_program_linking_api.py -m llm_api`
- Debug output: Set `LLMPROC_DEBUG=true` environment variable for detailed logging

### Test Coverage
- Unit tests: Cover core functionality without API calls (~41% code coverage)
- Provider tests: Mock API responses to test provider integrations
- MCP tests: Verify Model Context Protocol tool implementation
- Program Linking tests: Verify LLM-to-LLM communication features
- API Integration tests: Verify end-to-end functionality with real API calls
- All tests are marked appropriately to skip API tests by default

## Code Style Guidelines
- Uses absolute imports (`from llmproc import X`)
- Type hints on all functions and methods
- Google-style docstrings
- Max line length of 88 characters (Black)
- PEP8 compliant with Ruff enforcement

## LLMProcess Features
- Configurable via TOML files
- Supports system prompts from strings or files
- Maintains conversation state
- Parameters configurable via TOML
- File preloading for context (via [preload] section in TOML or preload_files() method)
- Custom display names for models in CLI interfaces
- Supports OpenAI, Anthropic, and Vertex AI models
- MCP (Model Context Protocol) support for tool usage
- Program linking for LLM-to-LLM communication via spawn tool
- Methods: run() [async], get_state(), reset_state(), from_toml(), preload_files()
- Command-line interface for interactive chat sessions
- Comprehensive error handling and diagnostics

## Session Summary (2025-03-12)
1. Restructured project to src/llmproc layout
2. Added comprehensive tests with pytest
3. Added type hints and documentation
4. Set up pre-commit hooks
5. Added dev tools configuration
6. Fixed file path issues in examples
7. Updated example script for shorter responses
8. Created and updated repository documentation

## Session Summary (2025-03-13)
1. Added file preloading feature to LLMProcess
2. Created preload.toml example to demonstrate the feature
3. Updated reference.toml with preload section documentation
4. Enhanced example.py to showcase preload functionality
5. Implemented XML-formatted warnings for missing files
6. Added XML-tagged format for preloaded file content
7. Updated reset_state to handle preloaded content consistently
8. Added preload_files() method for runtime file preloading
9. Created detailed documentation in docs/preload-feature.md
10. Added LLM evaluation test for preload feature
11. Updated pytest.ini with markers for API tests
12. Updated documentation in repo-map.txt and CLAUDE.md

## Session Summary (2025-03-14)
1. Added command-line interface (CLI) for interactive chat
2. Implemented model display name feature for better UX
3. Fixed Anthropic API system message handling 
4. Created configuration file selection functionality
5. Added configuration summary display
6. Implemented interactive chat with custom prompts
7. Added direct TOML file path specification support
8. Successfully merged CLI feature with preload functionality
9. Updated examples with display_name field
10. Updated documentation for CLI and display_name features

## Session Summary (2025-03-15)
1. Implemented async run method with proper tool execution support
2. Created unified API for both synchronous and asynchronous contexts
3. Added event loop detection and automatic handling
4. Fixed multi-turn tool execution for Anthropic models
5. Improved error handling for MCP tool responses
6. Added debug_tools parameter for detailed tool execution logging
7. Created comprehensive examples demonstrating tool usage
8. Updated documentation to reflect the new unified run method
9. Added unit tests for async tool execution
10. Updated README and feature status documentation

## Session Summary (2025-03-16)
1. Refactored provider implementations into a dedicated providers/ directory
2. Created separate anthropic_tools.py module for Anthropic-specific tool implementations
3. Fixed provider imports in test files for compatibility with new structure
4. Added debug_dumps directory for error logging and debugging
5. Fixed empty text blocks handling in Anthropic API messages
6. Completed repository cleanup and refactoring
7. Updated MCP configuration and documentation
8. Fixed test imports to work with new module structure
9. Updated repo-map.txt to reflect the new directory organization
10. Maintained backward compatibility for existing code

## Session Summary (2025-03-17)
1. Refactored MCP tools implementation for better organization and maintainability
2. Improved async handling in tool execution flow
3. Added proper type annotations throughout MCP code
4. Separated tool response processing from tool execution logic
5. Implemented more robust error handling for Anthropic API
6. Added specific tests for MCP tools functionality
7. Created mock time server test infrastructure
8. Updated pytest.ini to properly support async tests
9. Improved test isolation to avoid requiring API keys
10. Updated documentation in CONTRIBUTING.md and CLAUDE.md

## Session Summary (2025-03-18)
1. Completely rewrote _initialize_mcp_tools for clarity and efficiency
2. Implemented flexible tool name matching with case-insensitivity
3. Added support for different naming conventions (snake_case/camelCase)
4. Fixed proper namespacing of tools with server prefixes
5. Improved server configuration validation and error handling
6. Removed unpredictable automatic fallback behaviors
7. Added better diagnostic messages for tool registration
8. Created comprehensive tests for MCP tools functionality
9. Fixed mcp.toml example to use correct tool naming format
10. Optimized server initialization by checking configuration first

## Session Summary (2025-03-19)
1. Implemented program linking feature for communication between LLMs
2. Created spawn tool for delegating queries to specialized LLMs
3. Added [linked_programs] section to TOML configuration
4. Added support for [tools] section to enable built-in tools
5. Improved parameter handling with cleaner API parameters approach
6. Implemented custom tool handlers for non-MCP tools
7. Added example configurations in examples/program_linking/
8. Created detailed documentation in docs/program-linking.md
9. Added comprehensive tests for program linking functionality
10. Updated reference.toml with documentation for new sections

## Session Summary (2025-03-20)
1. Fixed critical bug in program linking with Anthropic API error handling
2. Removed unnecessary filter_empty_text_blocks function in favor of direct filtering
3. Fixed empty message handling to prevent 400 errors from Anthropic API
4. Updated LLMProcess to skip empty messages when preparing API requests
5. Modified run_anthropic_with_tools to only add messages with content
6. Added fallback message for empty responses
7. Created comprehensive test cases for empty message handling
8. Added robust test file for program linking unit tests
9. Created API integration tests for program linking with real APIs
10. Updated CLAUDE.md with detailed testing procedures
11. Added MISC.md with additional documentation and advanced usage
12. Added LLMPROC_DEBUG environment variable for detailed debugging