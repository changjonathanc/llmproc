# API Testing Guide for LLMProc

This guide covers all aspects of API testing in the LLMProc project, combining guidance on running tests and standardized patterns for writing them.

## Test Tiers

API tests are organized into three tiers based on their scope and execution time:

1. **Essential API Tests** (`essential_api`)
   - Minimal tests for CI/CD pipelines and daily development
   - Fast execution (~8-10 seconds)
   - Use the smallest models (Claude Haiku, GPT-4o-mini)
   - Very low token limits (20-50)
   - Simple prompts
   - Focus on core functionality only

2. **Extended API Tests** (`extended_api`)
   - Medium coverage for regular validation
   - Reasonable execution time (~15-20 seconds)
   - Use small but capable models
   - Medium token limits (50-150)
   - More comprehensive testing scenarios

3. **Release API Tests** (`release_api`)
   - Comprehensive coverage for pre-release validation
   - May take longer to run (~30-60+ seconds)
   - Include edge cases and complex interactions
   - Use target production models (when needed)
   - Higher token limits

## Running API Tests

### Using the `run_api_tests.py` Script

The included script provides a convenient way to run different test tiers:

```bash
# Run essential tests (fastest)
python tests/run_api_tests.py --tier essential

# Run extended tests (medium coverage)
python tests/run_api_tests.py --tier extended

# Run release tests (comprehensive)
python tests/run_api_tests.py --tier release

# Run all tests
python tests/run_api_tests.py --tier all
```

Additional options:
- `--workers N`: Number of parallel workers (default: 2)
- `--verbose`: Enable verbose output
- `--provider PROV`: Only run tests for a specific provider (anthropic, openai, vertex)
- `--coverage`: Generate coverage report

### Using pytest Directly

You can also use pytest directly for more control:

```bash
# Run essential API tests
pytest --run-api-tests -m "essential_api"

# Run extended API tests
pytest --run-api-tests -m "extended_api"

# Run release API tests
pytest --run-api-tests -m "release_api"

# Run all API tests
pytest --run-api-tests

# Run tests for a specific provider
pytest --run-api-tests -m "anthropic_api"
```

## Test Requirements

All API tests require the following:

1. API keys set in environment variables:
   - `ANTHROPIC_API_KEY` for Anthropic tests
   - `OPENAI_API_KEY` for OpenAI tests
   - `GOOGLE_APPLICATION_CREDENTIALS` for Vertex AI tests

2. The `--run-api-tests` flag to allow tests to make actual API calls

## Test Categories

1. **Unit Tests**:
   - Test individual components in isolation
   - No external dependencies or API calls
   - Fast, deterministic results
   - Located in `tests/unit/` directory

2. **Integration Tests (Non-API)**:
   - Test interactions between components
   - Mock external APIs but test real internal interactions
   - Located in main `tests/` directory

3. **API Tests**:
   - Test with real LLM API calls
   - Marked with `@pytest.mark.llm_api` and provider-specific markers
   - Include timing checks with reasonable timeouts
   - Located in main `tests/` directory

4. **Configuration Tests**:
   - Test loading and validating configurations
   - Focus on TOML parsing and validation
   - No process instantiation or API calls
   - Located in main `tests/` directory

## Standard Test Patterns

### 1. API Test Pattern (Fixture-based)

```python
import pytest
from tests.patterns import timed_test, assert_successful_response
from tests.conftest_api import claude_process_with_tools

@pytest.mark.llm_api
@pytest.mark.anthropic_api
@pytest.mark.essential_api
@pytest.mark.asyncio
async def test_feature_name(claude_process_with_tools):
    """Test description with clear purpose."""
    # Arrange
    process = claude_process_with_tools
    prompt = "Test prompt"

    # Act
    with timed_test(timeout_seconds=8.0):
        result = await process.run(prompt)

    # Assert
    assert_successful_response(result)
    assert "Expected output" in process.get_last_message()
```

### 2. Unit Test Pattern (Mock-based)

```python
import pytest
from unittest.mock import patch, MagicMock

from llmproc.module import function_to_test

def test_function_unit():
    """Test description."""
    # Arrange
    mock_dependency = MagicMock()

    # Act
    with patch("llmproc.module.dependency", mock_dependency):
        result = function_to_test("input")

    # Assert
    assert result == "expected"
    mock_dependency.assert_called_once_with("input")
```

### 3. Function-Based API Tests with Fixtures (Preferred Approach)

```python
import pytest
from tests.patterns import timed_test

@pytest.fixture
async def configured_process(claude_base_process):
    """Setup process with file descriptor enabled."""
    process = claude_base_process
    process.file_descriptor_enabled = True
    yield process

@pytest.mark.llm_api
@pytest.mark.anthropic_api
@pytest.mark.essential_api
@pytest.mark.asyncio
async def test_specific_behavior(configured_process):
    """Test specific behavior."""
    # Arrange
    process = configured_process

    # Act
    with timed_test(timeout_seconds=8.0):
        result = await process.run("Test input")

    # Assert
    assert result.is_success
    assert "Expected output" in process.get_last_message()

### 4. Class-Based Tests (Use Only When Necessary)

```python
import pytest
from tests.patterns import timed_test

@pytest.mark.llm_api
@pytest.mark.anthropic_api
class TestComplexFeature:
    """Test suite for complex feature requiring shared setup."""

    @pytest.fixture(autouse=True)
    async def setup(self):
        """Set up complex test environment."""
        self.complex_setup = await create_complex_environment()
        yield
        await self.complex_setup.cleanup()

    @pytest.mark.asyncio
    async def test_scenario_one(self, claude_base_process):
        """Test first scenario using shared setup."""
        # Arrange
        process = claude_base_process

        # Act & Assert
        with timed_test():
            # Test using self.complex_setup
            pass
```

Note: Class-based tests should only be used when there's significant shared setup that would be cumbersome with fixtures, or when testing many interrelated methods of a complex class.

### 5. Configuration Test Pattern

```python
import pytest
import tempfile
from pathlib import Path

from llmproc import LLMProgram

def test_config_validation(tmp_path: Path):
    """Test configuration validation."""
    # Arrange
    config_file = tmp_path / "test.toml"
    config_file.write_text("""
    [model]
    name = "test-model"
    provider = "test-provider"
    [prompt]
    system_prompt = "Test prompt"
    """)

    # Act
    program = LLMProgram.from_toml(config_file)

    # Assert
    assert program.model_name == "test-model"
    assert program.provider == "test-provider"
    assert program.system_prompt == "Test prompt"
```

## Fixtures and Utilities

### Standard Fixtures

1. **Program Fixtures** (session-scoped):
   - `base_program`: Basic program with minimal configuration
   - `claude_program`, `openai_program`, etc.: Provider-specific programs
   - Used to create process fixtures

2. **Process Fixtures** (function-scoped):
   - `mocked_llm_process`: Process with mocked API calls
   - `minimal_claude_process`, `minimal_openai_process`, etc.: Basic processes
   - `claude_process_with_tools`, etc.: Feature-specific processes

3. **Helper Fixtures**:
   - `tmp_path`: pytest-provided temporary directory
   - `create_test_process`: Function to create test processes properly

### Utility Functions

These are available in `tests/patterns.py`:

1. `timed_test(timeout_seconds=8.0)`: Context manager for timing checks
2. `assert_successful_response(result)`: Verify successful responses
3. `assert_error_response(result, error_text=None)`: Verify error responses

## Writing API Tests

When writing API tests, please follow these guidelines:

1. **Add appropriate markers**:
   ```python
   @pytest.mark.llm_api  # Required for all API tests
   @pytest.mark.essential_api  # Or extended_api or release_api
   @pytest.mark.anthropic_api  # Or openai_api or vertex_api
   ```

2. **Use optimized test patterns**:
   - Use smaller models (CLAUDE_SMALL_MODEL, OPENAI_SMALL_MODEL constants)
   - Set low max_tokens limits (20-50 for essential, 50-150 for extended)
   - Keep system prompts simple
   - Add timing checks to ensure tests complete within expected timeframes

3. **Add timing assertions**:
   ```python
   # Start timing
   start_time = time.time()

   # Test logic here...

   # Check timing
   duration = time.time() - start_time
   assert duration < 10.0, f"Test took too long: {duration:.2f}s > 10.0s timeout"
   ```

4. **Use session-scoped fixtures** for expensive operations:
   ```python
   @pytest.fixture(scope="session")
   def shared_resource():
       # Expensive setup here
       resource = setup_expensive_resource()
       yield resource
       # Cleanup here
   ```

5. **Provide clear test descriptions** in docstrings to explain purpose and expected behavior

## Best Practices

1. **Follow Arrange-Act-Assert (AAA) Pattern**:
   - Clearly separate setup, action, and verification
   - Add comments for each section: `# Arrange`, `# Act`, `# Assert`

2. **Use Fixtures for Common Setup**:
   - Move repeated setup into fixtures
   - Use function scope for process fixtures to ensure isolation
   - Use session scope for program fixtures to improve performance

3. **Include Timing Checks for API Tests**:
   - Use `timed_test` context manager
   - Set reasonable timeouts (8 seconds is usually sufficient)

4. **Use Descriptive Test Names**:
   - Name tests with `test_` prefix
   - Include what's being tested and expected behavior
   - Example: `test_calculator_handles_division_by_zero`

5. **Add Detailed Docstrings**:
   - Explain the purpose of the test
   - Document any specific requirements or edge cases
   - Include Args/Yields sections for fixtures

6. **Mark Tests Appropriately**:
   - `@pytest.mark.llm_api`: All API tests
   - `@pytest.mark.anthropic_api`, `@pytest.mark.openai_api`, etc.: Provider-specific
   - `@pytest.mark.essential_api`, `@pytest.mark.extended_api`, etc.: Test tiers

7. **Isolate Tests**:
   - Each test should be independent and self-contained
   - Reset state between tests
   - Don't rely on test execution order

8. **Use Parameterized Tests for Similar Cases**:
   - `@pytest.mark.parametrize` for testing multiple inputs
   - Reduces duplication in test code
   - Makes adding new test cases easier

## Debugging and Troubleshooting

Common issues:

1. **Test deselection**: Make sure you're using both the marker AND `--run-api-tests` flag
2. **Timeouts**: Check for resource-intensive operations or overusing large models
3. **API limits**: Reduce token counts and use smaller models when possible

## Migrating Existing Tests

If you're updating existing tests to follow these patterns:

1. Replace direct `LLMProcess()` calls with `create_test_llmprocess_directly`
2. Replace `create_test_llmprocess_directly` with standard process fixtures
3. Add appropriate test markers
4. Structure tests with Arrange-Act-Assert pattern
5. Replace custom timing code with `timed_test`
6. Replace common assertions with helpers from `patterns.py`
7. Update docstrings to follow the standard format
