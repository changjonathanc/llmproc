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
- Start CLI: `llmproc-demo ./examples/mcp.toml`
- Run tests: `pytest -v`
- Manage dependencies: `uv add <package>`, `uv add --dev <package>`, `uv remove <package>`
- Create worktree: `git worktree add worktrees/feature-name feature/feature-name`

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
- MCP (Model Context Protocol) support for tool usage (in development)
- Methods: run() [async], get_state(), reset_state(), from_toml(), preload_files()
- Command-line interface for interactive chat sessions

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