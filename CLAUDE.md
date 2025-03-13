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
- Install: `uv pip install -e .`
- Run example: `python example.py`
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
- File preloading for context (via [preload] section in TOML)
- Supports OpenAI, Anthropic, and Vertex AI models
- Methods: run(), get_state(), reset_state(), from_toml()

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
8. Updated documentation in repo-map.txt and CLAUDE.md