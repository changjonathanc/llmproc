# Contributing to LLMProc

Thank you for considering contributing to LLMProc! This document provides guidelines and instructions for contributing.

## Development Environment

### Setup

1. Fork and clone the repository
2. Set up your development environment:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode with all optional dependencies
uv pip install -e ".[dev,all]"

# Alternative: If you're working on a specific feature, you can install only what you need
uv pip install -e "."                  # Base package only
uv pip install -e ".[dev,openai]"      # Development + OpenAI
uv pip install -e ".[dev,anthropic]"   # Development + Anthropic

# Sync the virtual environment with pyproject.toml changes
uv pip sync

# Install pre-commit hooks
pre-commit install
```

If you need to update your environment after pulling changes that modify dependencies:

```bash
# Sync environment with latest dependencies
uv pip sync

# Or reinstall with specific extras
uv pip install -e ".[dev,all]"
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

# Add a provider-specific optional dependency
uv add --optional openai package_name
uv add --optional anthropic package_name
uv add --optional vertex package_name
uv add --optional gemini package_name
# This adds the dependency to the respective [project.optional-dependencies] section

# Remove a dependency
uv remove package_name

# Update lockfile after dependency changes
uv lock

# The uv.lock file should be committed to git to ensure reproducible builds
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
- Write docstrings for all modules, classes, and public methods/functions
  - Internal methods can omit docstrings if their purpose is simple and obvious
  - Use separate `_docs_*.py` files only when:
    - A significant portion of a file would be docstrings, and
    - The file is already long (>400 lines)
  - For staticmethods, always use native docstrings to avoid Python 3.12+ compatibility issues
- Use absolute imports (`from llmproc import X` instead of relative imports)
- Maximum line length is 200 characters
- Format code with Ruff before submitting PRs
  ```bash
  # Run Ruff linter
  ruff check .

  # Apply automatic fixes
  ruff check --fix .

  # Format code
  ruff format .
  ```

## Git Workflow

The project includes a `.gitconfig` file that configures Git to create merge commits by default (no fast-forward merges). This preserves branch history and makes the repository history more navigable.

Key Git guidelines:
- Create a branch for each feature or bugfix
- Rebase your branch onto main before creating a PR
- Always use merge commits (not fast-forward) when merging PRs
- Use descriptive commit messages explaining why changes were made
- Squash fixup commits before merging using `git rebase -i main`

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

### Working with Git Worktrees

When working in a git worktree, it's recommended to create an isolated virtual environment within that worktree to avoid conflicts with other worktrees or the main repository:

```bash
# Navigate to the worktree directory
cd worktrees/feature-name

# Create a worktree-specific virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode
uv pip install -e ".[dev,all]"

# Now you can run tests within the isolated environment
pytest tests/test_specific_feature.py
```

This approach ensures that each worktree has its own isolated dependencies and prevents conflicts when running tests or developing features simultaneously across multiple worktrees.

Note: Tests marked with `@pytest.mark.llm_api` require actual API keys and make network calls to LLM providers. These are skipped by default to allow running tests in CI environments without API keys.

## Pull Request Process

1. Create a new branch for your feature or bugfix
2. Write tests for your changes
3. Ensure all tests pass and code is properly formatted
4. Update documentation if needed
5. Submit a pull request

### Git Merge Strategy

When merging pull requests, prefer the standard merge commit approach over squash merges:

```bash
# Preferred: Standard merge
git merge --no-ff feature-branch

# Avoid: Squash merge
git merge --squash feature-branch
```

Reasons to avoid squash merges:
- They break Git's ability to track which branches have been fully merged
- This makes branch cleanup more difficult, as `git branch --merged` can't identify squashed branches
- It loses the detailed commit history from the feature branch
- Makes it harder to revert specific changes from a feature

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
