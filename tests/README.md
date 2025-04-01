# LLMProc Testing Guide

This directory contains tests for the LLMProc framework. The tests are organized by feature and type, following pytest conventions.

## Test Organization

Tests are organized into several categories:

1. **Unit Tests**: Test individual components in isolation (no API calls)
2. **Integration Tests**: Test interactions between components (no API calls)
3. **API Tests**: Tests that require actual API calls to LLM providers
4. **CLI Tests**: Tests for the command-line interface
5. **Example Tests**: Tests that verify example configurations

## Running Tests

### Basic Test Run (No API Calls)

```bash
# Run all non-API tests
pytest

# Run specific test file
pytest tests/test_file.py

# Run specific test in a file
pytest tests/test_file.py::test_function
```

### Running API Tests

API tests require valid API keys for the respective LLM providers:
- `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` for Anthropic/Claude tests
- `OPENAI_API_KEY` for OpenAI tests

```bash
# Run all tests including API tests
pytest -m "llm_api"

# Run a specific API test file
pytest tests/test_file.py -m "llm_api"
```

## Test Naming Conventions

- `test_*.py`: All test files
- `test_*_integration.py`: Integration tests
- `test_*_api.py`: Tests requiring real API calls
- `test_example_*.py`: Tests for example configurations

## Test Categories

| Category | Prefix/Suffix | Description | Example |
|----------|---------------|-------------|---------|
| Core | `test_llm_process*.py` | Tests of the core LLMProcess functionality | `test_llm_process.py` |
| Providers | `test_*_process_executor.py` | Tests for specific providers | `test_anthropic_process_executor.py` |
| Tools | `test_*_tool.py` | Tests for specific tools | `test_calculator_tool.py` |
| Program Linking | `test_program_linking*.py` | Tests for program linking | `test_program_linking.py` |
| File Descriptor | `test_file_descriptor*.py` | Tests for file descriptor system | `test_file_descriptor.py` |
| CLI | `test_cli*.py` | Tests for command-line interface | `test_cli.py` |
| Reasoning Models | `test_*_reasoning_models*.py` | Tests for reasoning models | `test_openai_reasoning_models.py` |
| Configuration | `test_from_toml.py` | Tests for TOML configuration loading | `test_from_toml.py` |

## Test Markers

- `@pytest.mark.llm_api`: Tests that make actual API calls to LLM providers
- `@pytest.mark.asyncio`: Tests that use asyncio functionality

## Adding New Tests

When adding new tests, follow these guidelines:

1. Use appropriate naming convention based on test type
2. Mark API tests with `@pytest.mark.llm_api`
3. Include graceful skipping for missing API keys in API tests
4. Follow existing patterns for similar functionality
5. Include both positive and negative test cases
6. Aim for high test coverage of core functionality

## Future Test Improvements

See [test-plan.md](../docs/test-plan.md) for the comprehensive test plan and future improvements.