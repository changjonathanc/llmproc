# LLMProc Strategic Testing Plan

This document outlines the strategic approach to testing in the LLMProc project, with a focus on minimizing API usage costs while maintaining comprehensive test coverage.

## Core Testing Philosophy

1. **Minimize Live API Calls**: Use mock tests wherever possible, reserving actual API calls for critical functionality tests.
2. **Isolate Test Process Instances**: Never reuse process instances between tests to ensure test isolation and prevent state contamination.
3. **Optimize Test Tiers**: Structure tiers to ensure coverage without redundancy, where extended tier should provide exhaustive functional coverage.
4. **Strategic Model Selection**: Use smaller, faster, and cheaper models (e.g., Claude Haiku, GPT-4o-mini) for testing whenever possible.

## Test Tiers Redefined

### Tier 1: Essential API Tests
- **Purpose**: Daily development validation and CI/CD pipelines
- **Coverage**: Core API functionality only
- **Execution Time**: < 30 seconds total
- **Model Usage**: Smallest available models with minimal token counts
- **Characteristics**:
  - Focused on single-purpose tests that validate core behavior
  - No redundancy between tests
  - All tests have strict timeouts (< 5 seconds per test)
  - Extremely cost-efficient

### Tier 2: Extended API Tests
- **Purpose**: Comprehensive functional coverage before merging PRs
- **Coverage**: All functional capabilities with representative tests
- **Execution Time**: < 2 minutes total
- **Model Usage**: Small models with reasonable token counts
- **Characteristics**:
  - Ensures all features work correctly
  - Covers edge cases and error handling
  - Each feature has at least one representative test
  - **IMPORTANT**: If all extended tests pass, a PR should be functionally correct and ready to merge

### Tier 3: Release API Tests
- **Purpose**: Configuration validation and compatibility verification
- **Coverage**: Example configurations and file validation
- **Execution Time**: < 5 minutes total
- **Model Usage**: Production models when required by examples
- **Characteristics**:
  - Validates all example TOML files
  - Ensures existing configurations continue to work
  - Checks for syntax errors or outdated patterns in examples
  - NOT focused on functional testing (which is handled by Extended tier)

## Mock Testing Strategy

### CLI Tests
- Most CLI tests should use mocks instead of actual API calls
- Only test CLI-specific functionality with actual APIs when needed
- Focus on interface validation, not model response validation

```python
# Example of mocking for CLI tests
@patch("llmproc.llm_process.LLMProcess.run")
def test_cli_functionality(mock_run):
    # Mock the run method to return a predefined response
    mock_run.return_value = RunResult()
    mock_run.return_value.set_response("Mocked response")
    
    # Test CLI with the mock
    result = subprocess.run(["llmproc-demo", "config.toml", "-p", "test prompt"], 
                           capture_output=True, text=True)
    assert "Mocked response" in result.stdout
```

### Provider Tests
- Develop provider-specific mock fixtures
- Test provider-specific features in isolation
- Only use real API calls for validating client integration

```python
# Provider-specific mock for testing
@pytest.fixture
def mock_anthropic_client():
    with patch("anthropic.AsyncAnthropic") as mock_client:
        # Configure the mock to simulate API responses
        mock_client.return_value.messages.create.return_value = MagicMock(
            content=[{"type": "text", "text": "Mock response"}],
            stop_reason="end_turn"
        )
        yield mock_client
```

## Implementing De-Duplication

### Program Linking Tests

Current redundant test coverage:
1. `test_program_linking_api_optimized.py`: Core functionality tests (3 tests)
2. `test_example_programs.py`: Contains `test_program_linking_functionality()`
3. CLI tests: Contains `test_cli_with_program_linking()`

Recommendation:
- Keep `test_program_linking_api_optimized.py` tests in Extended tier
- Move `test_program_linking_functionality` from Essential to Release tier
- Keep CLI test with mocked API responses

### Example Program Tests

Current approach:
- Tests all example programs with actual API calls
- Very expensive and time-consuming

Recommended approach:
- Test a representative subset in Extended tier
- Test all examples in Release tier
- Use syntax validation without API calls where possible

```python
# Example of non-API validation for TOML files
def test_example_syntax():
    # Get all example programs
    example_files = glob.glob("examples/**/*.toml", recursive=True)
    
    # Validate each program loads without errors
    for example in example_files:
        try:
            program = LLMProgram.from_toml(example)
            # Validation happens during from_toml
            assert True, f"{example} loaded successfully"
        except Exception as e:
            pytest.fail(f"Failed to load {example}: {str(e)}")
```

## Test Suite Organization

### Proposed Structure

```
tests/
├── unit/               # No API calls
├── integration/        # No API calls
├── api/
│   ├── essential/      # Tier 1 API tests
│   ├── extended/       # Tier 2 API tests
│   └── release/        # Tier 3 API tests
├── mock/               # Mock fixtures and utilities
└── conftest.py         # Shared fixtures
```

### Implementation Status

### Completed Actions:
1. **Core Functionality**:
   - ✅ Updated `test_program_linking_api_optimized.py` to cover all necessary functionality
   - ✅ Removed redundant `test_program_linking_api.py` file
   - ✅ Added mock fixtures for all providers (Anthropic, OpenAI, Gemini)
   - ✅ Updated tier assignments in existing tests to avoid redundancy
   - ✅ Increased timeouts in API tests to accommodate real-world latency

2. **Mock Testing**:
   - ✅ Implemented provider-specific mock fixtures
   - ✅ Created example CLI tests with proper mocking patterns
   - ✅ Added documentation for CLI testing challenges
   - ✅ Created helper classes for simulating LLM responses

3. **Validation**:
   - ✅ Verified that essential API tests pass with current implementation
   - ✅ Verified that extended API tests pass with current implementation
   - ✅ Skipped redundant tests to prevent unnecessary API usage

### Pending Items:
1. **Short-term Improvements**:
   - Add syntax validation tests for example files
   - Enhance CLI test implementation
   - Implement more comprehensive mock coverage

2. **Long-term Strategy**:
   - Refactor test suite organization
   - Implement CI/CD pipeline that runs appropriate tiers
   - Add cost tracking for API tests

## Best Practices

1. **Test Isolation**:
   - Always create new process instances for each test
   - Use `async` fixtures with proper cleanup

```python
@pytest.fixture
async def isolated_process():
    # Create a fresh program and process for each test
    program = LLMProgram(...)
    process = await program.start()
    yield process
    # Clean up if necessary
```

2. **Timeout Control**:
   - Set explicit timeouts for all API tests
   - Use tiered timeout values based on test category
   - Monitor and update timeouts periodically

3. **Functional Coverage**:
   - Extended tier should cover ALL functional capabilities
   - If a test passes in Extended, it should pass in Release
   - Failures in Release should only be due to configuration issues, not core functionality

4. **Test Documentation**:
   - Clearly document the purpose of each test
   - Explain mock behavior and test assumptions
   - Document expected API usage for each test

5. **CLI Testing Challenges**:
   - CLI tests are particularly challenging to mock completely due to process isolation
   - Consider two complementary approaches:
     1. Use subprocess mocking to test CLI argument handling without any API calls
     2. Create simplified CLI tests that use real API calls but with minimal interactions
   - When possible, test CLI components separately from actual command-line interface

```python
# Example of a CLI integration point to facilitate testing
def create_and_start_process(config_path, **kwargs):
    """Centralized function that CLI and tests can both use."""
    program = LLMProgram.from_toml(config_path)
    process = await program.start()
    return process
```

## Monitoring and Optimization

- Implement API call counting during test runs
- Track cost metrics to identify expensive tests
- Regularly review and optimize the most expensive tests
- Consider implementing a budget limit for CI/CD test runs

By implementing this strategic approach, we can maintain comprehensive test coverage while minimizing API usage costs and ensuring efficient development workflows.