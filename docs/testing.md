# Testing Guide for LLMProc

This document outlines the testing approach, structure, and procedures for the LLMProc project.

## Testing Philosophy

The LLMProc project uses a multi-layered testing approach:

1. **Unit Tests**: Test individual components in isolation (no API calls)
2. **Integration Tests**: Test interactions between components
3. **API Tests**: Verify functionality with real LLM APIs (marked with `llm_api`)
4. **Program Tests**: Verify example programs work correctly

Tests are designed to work without requiring API credentials during regular development, with separate test suites that can be run to verify API integration when needed.

## Test Organization

### Test Files

- **test_llm_process.py**: Core LLMProcess functionality
- **test_llm_process_providers.py**: Provider-specific functionality
- **test_example_programs.py**: Tests for all example programs
- **test_from_toml.py**: Loading LLMProcess from TOML files
- **test_mcp_features.py**: MCP tool functionality
- **test_mcp_tools.py**: Specific MCP tool implementations
- **test_program_linking.py**: Basic program linking tests
- **test_program_linking_robust.py**: Robust program linking tests
- **test_program_linking_api.py**: API tests for program linking
- **test_providers.py**: Provider client initialization

### Test Markers

Tests are organized using pytest markers:

- `llm_api`: Tests that require API keys and make real API calls
- `asyncio`: Tests for asynchronous functionality

## Running Tests

### Basic Test Commands

```bash
# Run all tests (skips API tests)
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=src/llmproc

# Run specific test file
pytest tests/test_llm_process.py

# Run specific test
pytest tests/test_llm_process.py::test_initialization
```

### API Testing

API tests require valid API keys and are skipped by default. To run API tests:

```bash
# Set required environment variables
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
export GOOGLE_CLOUD_PROJECT=your_gcp_project  # For Vertex AI

# Run all API tests
pytest -m llm_api

# Run specific API test file
pytest -m llm_api tests/test_example_programs.py
```

See [API Testing Guide](api_testing.md) for detailed information on API tests.


## Test Coverage

The project maintains test coverage in these key areas:

1. **Core Functionality**:
   - LLMProcess initialization
   - Conversation state management
   - Reset functionality
   - Parameter handling

2. **Provider Integration**:
   - OpenAI provider tests
   - Anthropic provider tests
   - Vertex AI provider tests

3. **Feature Tests**:
   - File preloading
   - MCP tool integration
   - Program linking
   - Parameter handling

4. **Example Program Tests**:
   - Tests all example TOML programs
   - Verifies CLI functionality

## Writing New Tests

### Test Structure

Follow these guidelines when writing new tests:

1. **Use pytest fixtures** for setup and teardown
2. **Mock external dependencies** (especially API calls)
3. **Mark API tests** with `@pytest.mark.llm_api`
4. **Test error cases** as well as happy paths
5. **Follow the pattern** of existing tests for consistency

### Example Test

```python
import pytest
from unittest.mock import patch, MagicMock

from llmproc import LLMProcess

# Unit test that doesn't require API access
def test_parameter_handling():
    """Test that parameters are correctly passed to the API."""
    process = LLMProcess(
        model_name="test-model",
        provider="openai",
        system_prompt="Test system prompt",
        parameters={"temperature": 0.7}
    )

    assert process.api_params["temperature"] == 0.7

# API test that requires credentials
@pytest.mark.llm_api
async def test_actual_api_call():
    """Test an actual API call (requires API keys)."""
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OpenAI API key not available")

    process = LLMProcess(
        model_name="gpt-4o-mini",
        provider="openai",
        system_prompt="You are a test assistant."
    )

    response = await process.run("Say hello")
    assert response
    assert isinstance(response, str)
```

## Test Isolation

Tests are designed to be isolated and not depend on each other. Key isolation techniques:

1. **Environment variable management**: Tests restore original environment variables
2. **Mock external services**: API calls are mocked for non-API tests
3. **Temporary files**: Tests use temporary files/directories where needed
4. **State reset**: LLMProcess state is reset between tests

## Debugging Failed Tests

When tests fail, check:

1. **API credentials**: For API tests, ensure keys are valid
2. **Mock correctness**: Ensure mocks match the expected API signature
3. **Async handling**: Ensure async tests are correctly defined
4. **Assertion details**: Check exact assertion failure messages

## Continuous Integration

The test suite runs in CI with these settings:

1. **Regular tests** run on every PR and commit
2. **API tests** are skipped in CI to avoid requiring credentials
3. **Coverage reports** are generated to track test quality
