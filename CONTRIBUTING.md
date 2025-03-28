# Contributing to LLMProc

Thank you for considering contributing to LLMProc! This document provides guidelines and instructions for contributing.

## Development Environment

### Setup

1. Fork and clone the repository
2. Set up your development environment:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install the package in development mode
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev,all]"  # Install with all optional dependencies

# Install pre-commit hooks
pre-commit install
```

### Managing Dependencies

We use `uv` and `pyproject.toml` to manage dependencies:

```bash
# Add a runtime dependency
uv add package_name
# This adds the dependency to pyproject.toml

# Add a development dependency
uv add --dev package_name
# This adds the dependency to the [project.optional-dependencies] dev section

# Remove a dependency
uv remove package_name

# Update lockfile (recommended after dependency changes)
uv lock
```

Note: We don't use requirements.txt for dependency management. All dependencies should be defined in pyproject.toml.

## Design Principles

### Make the Right Thing the Easy Thing

We follow the principle of making the right thing the easy thing. This means:
- Features with no functional downsides should be enabled by default
- Performance optimizations should be opt-out rather than opt-in
- Users shouldn't need to know implementation details to get optimal results
- The library should handle as much complexity as possible on behalf of users

Example: Prompt caching is automatically enabled for all Anthropic models without requiring any configuration, while still allowing users to disable it if needed.

## Code Standards

- Follow [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Add type hints to all functions and methods
- Write docstrings for all modules, classes, methods, and functions
- Use absolute imports (`from llmproc import X` instead of relative imports)
- Run tests and formatting before submitting PRs

## Testing

Run tests with pytest:

```bash
# Run all tests except those requiring API keys
pytest

# Run all tests (including those that make actual API calls)
pytest -m "all"

# Run only tests requiring API keys
pytest -m "llm_api"

# Run with coverage report
pytest --cov=llmproc
```

Note: Tests marked with `@pytest.mark.llm_api` require actual API keys and make network calls to LLM providers. These are skipped by default to allow running tests in CI environments without API keys.

## Pull Request Process

1. Create a new branch for your feature or bugfix
2. Write tests for your changes
3. Ensure all tests pass and code is properly formatted
4. Update documentation if needed
5. Submit a pull request

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow