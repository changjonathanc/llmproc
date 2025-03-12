# Contributing to LLMProc

Thank you for considering contributing to LLMProc! This document provides guidelines and instructions for contributing.

## Development Environment

### Setup

1. Fork and clone the repository
2. Set up your development environment:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package in development mode
uv pip install -e .

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
```

### Managing Dependencies

We use `uv` to manage dependencies:

```bash
# Add a dependency
uv add pandas

# Add a development dependency
uv add --dev pytest

# Remove a dependency
uv remove pandas

# Update requirements.txt
uv pip freeze > requirements.txt
```

## Code Standards

- Follow [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Add type hints to all functions and methods
- Write docstrings for all modules, classes, methods, and functions
- Use absolute imports (`from llmproc import X` instead of relative imports)
- Run tests and formatting before submitting PRs

## Testing

Run tests with pytest:

```bash
pytest
```

For coverage:

```bash
pytest --cov=llmproc
```

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