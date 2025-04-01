# RFC020: Test Suite Improvements

## Summary

This RFC proposes several improvements to the test suite architecture to enhance maintainability, reduce API usage costs, and improve test reliability. These recommendations are based on experience refactoring the existing test suite and addressing issues with API-dependent tests.

## Motivation

The current test suite has several issues that limit its effectiveness:

1. **Excessive API usage**: Many tests make API calls unnecessarily, increasing costs and test time
2. **Brittle paths**: Tests use hardcoded paths that break when examples are reorganized
3. **Model version dependencies**: Tests reference specific model versions directly, requiring updates when models change
4. **Limited non-API test coverage**: Many components are only tested with actual API calls
5. **Inconsistent test isolation**: API tests are not consistently marked with the appropriate markers

## Proposal

We propose the following improvements to the test architecture:

### 1. Path Constants and Standard Structure

```python
# Define at the top of test files (or in a common test_utils.py)
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
FEATURES_DIR = EXAMPLES_DIR / "features"
ANTHROPIC_DIR = EXAMPLES_DIR / "anthropic"
OPENAI_DIR = EXAMPLES_DIR / "openai"
CLAUDE_CODE_DIR = EXAMPLES_DIR / "claude-code"
```

Benefits:
- Single point of maintenance when paths change
- Clear indication of directory structure
- Easy to update all tests when example locations change

### 2. Model Version Constants

```python
# Define at the top of test files or in a constants.py file
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
OPENAI_GPT4O_MODEL = "gpt-4o-2024-05-13"
OPENAI_O3_MODEL = "o3-mini"
```

Benefits:
- Simplified updates when model versions change
- Consistent model versions across tests
- Clear indication of which models are being used

### 3. Split Configuration Tests from API Tests

For each test module, split into two test categories:
1. **Configuration Tests**: Test parameter parsing, validation, and transformation (no API calls)
2. **API Tests**: Test actual API integration marked with `@pytest.mark.llm_api`

Example:

```python
# Configuration test (no API marker)
def test_thinking_parameters_validation():
    program = LLMProgram.from_toml(ANTHROPIC_DIR / "claude-3-7-thinking-high.toml")
    assert program.parameters["thinking"]["type"] == "enabled"
    # More validation...

# API test with marker
@pytest.mark.llm_api
async def test_thinking_api_functionality():
    process = await program.start()
    result = await process.run("Test prompt")
    # API-dependent assertions...
```

Benefits:
- Clear separation between configuration and API tests
- Increased test coverage without API access
- Reduced API costs and test time

### 4. Mock-Based Parameter Transformation Tests

Use mocks to test parameter transformation logic without making API calls:

```python
def test_parameter_transformation():
    # Create program
    program = LLMProgram.from_toml(example_path)
    
    # Mock the executor to capture transformed parameters
    mock_executor = MagicMock()
    mock_executor._prepare_api_params.return_value = {...}
    
    # Test parameter transformation
    transformed_params = mock_executor._prepare_api_params(program.parameters)
    assert transformed_params["key"] == expected_value
```

Benefits:
- Test parameter transformation without API calls
- Isolate testing of transformation logic
- Identify transformation issues before API calls

### 5. Parameter Validation Helper Functions

Create helper functions to validate parameters in a structured way:

```python
def validate_thinking_parameters(program):
    """Validate thinking model parameters."""
    assert program.model_name.startswith("claude")
    assert "thinking" in program.parameters
    assert program.parameters["thinking"]["type"] == "enabled"
    # More validations...
    return program.parameters["thinking"]["budget_tokens"]
```

Benefits:
- Reusable validation logic
- Consistent parameter validation
- Better error messages

### 6. Test Documentation Standards

Standardize comments in test files:
- Document the purpose of each test class and method 
- Explain test dependencies and requirements
- Document why certain approaches were taken

Example:
```python
"""
Test suite for thinking models.

Configuration tests verify parameter structure and validation.
API tests verify actual model behavior with API calls.

Note: API calls require ANTHROPIC_API_KEY environment variable.
"""
```

Benefits:
- Clear understanding of test purpose
- Easier maintenance
- Knowledge transfer to new developers

## Migration Plan

1. Create a test_utils.py file with shared constants and helper functions
2. Update one test file at a time, starting with most frequently run tests
3. Add missing configuration tests for areas only covered by API tests
4. Update documentation with test organization guidelines

## Alternatives Considered

1. **Test Fixtures**: Using fixtures instead of helper functions. This approach would work but makes tests harder to follow.
2. **Test Class Inheritance**: Creating base test classes for common validation. This could lead to complex inheritance hierarchies.
3. **Environment-Based Test Selection**: Using environment variables to determine test depth. This adds complexity to test configuration.

## Open Questions

1. Should path constants be centralized in a single file or defined in each test module?
2. How should we handle version-specific tests as model behaviors change?
3. Should we create a versioned model registry to track model capabilities and limitations?

## Implementation Checklist

- [ ] Create test_utils.py with path constants
- [ ] Define model version constants
- [ ] Update test markers for clearer separation of API tests 
- [ ] Add helper functions for parameter validation
- [ ] Update CI configuration to run non-API tests by default
- [ ] Document testing approach in CONTRIBUTING.md