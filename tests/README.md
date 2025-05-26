# LLMProc Testing Guide

This directory contains tests for the LLMProc framework. The tests are organized by feature and type, following pytest conventions.

## Test Organization

Tests are organized into several categories:

1. **Unit Tests**: Test individual components in isolation (no API calls)
   - **Directory-based:** Located in `tests/unit/` directory for true unit tests with full isolation
   - **Suffix-based:** Files named `test_*_unit.py` for component-level unit tests

2. **Integration Tests**: Test interactions between components (no API calls)
   - Located in main `tests/` directory with `*_integration.py` suffix

3. **API Tests**: Tests that require actual API calls to LLM providers
   - Marked with `@pytest.mark.llm_api` and organized by tier

4. **CLI Tests**: Tests for the command-line interface
   - Located in `tests/cli/` directory with `test_cli_*.py` naming

5. **Example Tests**: Tests that verify example configurations and demo scripts
   - Located in `tests/examples/` directory

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
- `GOOGLE_APPLICATION_CREDENTIALS` for Vertex AI tests

#### Using the API Test Runner

```bash
# Run essential API tests (fastest, for CI and daily development)
python tests/run_api_tests.py --tier essential

# Run extended API tests (balanced coverage)
python tests/run_api_tests.py --tier extended

# Run release API tests (comprehensive validation for releases)
python tests/run_api_tests.py --tier release
```

#### Using pytest directly

API tests are marked with `@pytest.mark.llm_api` and are skipped by default to prevent accidental API usage. To run them:

```bash
# Run all API tests
pytest --run-api-tests

# Run only tests with specific markers
pytest --run-api-tests -m "essential_api"
pytest --run-api-tests -m "extended_api"
pytest --run-api-tests -m "release_api"

# Run a specific API test file
pytest --run-api-tests tests/test_file.py

# Combine with other pytest options
pytest --run-api-tests -xvs tests/test_file.py::test_function
```

## Test Documentation

The testing strategy is documented in several files:

- [**API_TESTING.md**](./API_TESTING.md): Comprehensive API testing guide with patterns and best practices
- [**TEST_STRATEGY.md**](./TEST_STRATEGY.md): Strategic approach to testing and test reorganization progress
- [**LLM_API_TESTS_SUMMARY.md**](./LLM_API_TESTS_SUMMARY.md): Detailed inventory of all API-dependent tests

## Test Naming Conventions

- `test_*.py`: All test files
- `test_*_integration.py`: Integration tests
- `test_*_api.py`: Tests requiring real API calls
- `test_example_*.py`: Tests for example configurations
- `examples/test_*.py`: Tests for example scripts and demo features

## Test Categories

| Category | Prefix/Suffix | Description | Example |
|----------|---------------|-------------|---------|
| Core | `test_llm_process*.py` | Tests of the core LLMProcess functionality | `test_llm_process.py` |
| Providers | `test_*_process_executor.py` | Tests for specific providers | `test_anthropic_process_executor.py` |
| Tools | `test_*_tool.py` | Tests for specific tools | `test_calculator_tool.py` |
| Program Linking | `test_program_linking*.py` | Tests for program linking | `test_program_linking_core.py` |
| File Descriptor | `test_file_descriptor*.py` | Tests for file descriptor system | `test_file_descriptor.py` |
| CLI | `cli/test_cli*.py` | Tests for command-line interface | `cli/test_cli.py` |
| Reasoning Models | `test_reasoning_models*.py` | Tests for reasoning models | `test_reasoning_models.py` |
| Configuration | `test_from_toml.py` | Tests for TOML configuration loading | `test_from_toml.py` |
| Examples | `examples/test_*.py` | Tests for example scripts and demos | `examples/test_goto_context_compaction.py` |

## Test Markers

### Basic Markers
- `@pytest.mark.llm_api`: Tests that make actual API calls to LLM providers
- `@pytest.mark.asyncio`: Tests that use asyncio functionality

### API Test Tier Markers
- `@pytest.mark.essential_api`: Essential API tests for CI and daily development (fastest)
- `@pytest.mark.extended_api`: Extended API tests for regular validation (balanced)
- `@pytest.mark.release_api`: Complete API tests for release validation (most comprehensive)

### Provider-Specific Markers
- `@pytest.mark.anthropic_api`: Tests specific to Anthropic's API
- `@pytest.mark.openai_api`: Tests specific to OpenAI's API
- `@pytest.mark.gemini_api`: Tests specific to Google Gemini API
- `@pytest.mark.vertex_api`: Tests specific to Vertex AI

## Adding New Tests

When adding new tests, follow these guidelines:

1. Use appropriate naming convention based on test type
2. For API tests:
   - Mark all API tests with `@pytest.mark.llm_api`
   - Add appropriate tier marker (`essential_api`, `extended_api`, or `release_api`)
   - Add provider-specific marker if applicable
3. Include graceful skipping for missing API keys in API tests
4. Follow existing patterns for similar functionality
5. Include both positive and negative test cases
6. Use the optimized fixtures in `conftest_api.py` for API tests
7. Add explicit timeout assertions for API tests
8. Use smaller models and minimal token settings when possible
9. Aim for high test coverage of core functionality
10. For demo/example tests, place them in the `tests/examples/` directory

## Test Reorganization Progress

We are currently in the process of reorganizing and consolidating tests:

- **Standardized Test Suite Structure**: ✅ Completed
- **Test Suite Structure Followup**:
  - **Stage 1**: ✅ Completed
  - **Stage 2**: 🔄 In Progress
  - **Stage 3**: 🔜 Planned

Recent progress includes:
- Created `tests/examples/` directory for demo-specific tests
- Moved `test_goto_context_compaction.py` to examples directory
- Consolidated many redundant test files (3,400+ lines removed)
- Updated documentation to reflect new organization

See [TEST_STRATEGY.md](./TEST_STRATEGY.md) for detailed information on test reorganization status and plans.
