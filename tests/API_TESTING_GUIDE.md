# API Testing Guide for llmproc

This guide explains the tiered testing approach for API tests in the llmproc library, as implemented in RFC027.

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

## Test Organization

- All API tests should be in files with names starting with `test_`
- Tests for specific providers should have their interfaces clearly defined
- Core integration tests should work with any provider (prefer Claude Haiku for speed)

## Debugging and Troubleshooting

Common issues:

1. **Test deselection**: Make sure you're using both the marker AND `--run-api-tests` flag
2. **Timeouts**: Check for resource-intensive operations or overusing large models
3. **API limits**: Reduce token counts and use smaller models when possible

For more information, see RFC027 on test suite improvements.