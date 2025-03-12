# CLAUDE.md - Session and Repository Summary

## Repository Structure
- `src/llmproc/`: Main package directory containing LLMProcess implementation
- `examples/`: TOML configuration examples (minimal.toml, complex.toml)
- `prompts/`: Contains system prompt templates for LLM configuration
- `tests/`: Test files with 90% coverage of codebase

## Key Commands
- Install: `uv pip install -e .`
- Run example: `python example.py`
- Run tests: `pytest -v`
- Manage dependencies: `uv add <package>`, `uv add --dev <package>`, `uv remove <package>`

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
- Supports OpenAI models
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