# CLAUDE.md - Session and Repository Summary

## Repository Structure
- `src/llmproc/`: Main package directory containing LLMProcess implementation
- `examples/`: TOML configuration examples (minimal.toml, complex.toml)
- `prompts/`: Contains system prompt templates for LLM configuration
- `tests/`: Test files with 90% coverage of codebase
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
- Debug output: Configure Python logging for `llmproc` package

### Test Coverage
- Unit tests: Cover core functionality without API calls (~41% code coverage)
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
- Max line length of 88 characters (Black)
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
- Program compiler for validation and preprocessing of configurations
- Methods: run() [async], get_state(), reset_state(), from_toml(), preload_files()
- Command-line interface for interactive chat sessions
- Comprehensive error handling and diagnostics

## Note on Session Summaries
Previous session summaries have been moved to the `session_summaries/` directory and are no longer tracked in version control.