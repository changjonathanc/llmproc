# Example/Demo Tests

This directory contains tests specifically for validating example scripts and demo features that showcase LLMProc capabilities to users.

## Purpose

These tests serve a different purpose from the core functionality tests:

1. **Documentation Validation:** They verify that the examples in documentation work correctly
2. **Demo Script Testing:** They test the example scripts in `examples/scripts/` directory
3. **End-to-End Feature Demos:** They validate complete user-facing workflows

## Current Example Tests

- **goto_context_compaction_test.py:** Tests the `goto_context_compaction_demo.py` script that demonstrates efficient context management
- **test_multiply_example.py:** Tests the `multiply_example.py` script that shows basic Python API usage

## Test Organization

Example tests should follow these principles:

1. Focus on end-to-end behavior rather than implementation details
2. Validate the user-facing aspects of the examples
3. Use minimal test fixtures and easy-to-understand assertions
4. Keep assertions focused on the example's primary purpose
5. Run with actual API calls when needed to verify real-world behavior

## Adding New Example Tests

When adding a new example test:

1. Name the test file to clearly indicate the example it's testing
2. Include both positive and negative test cases where appropriate
3. Add appropriate markers for API tests (`essential_api`, `extended_api`, etc.)
4. Document the purpose of the test at the top of the file
5. Use `@pytest.mark.llm_api` for tests that require API calls

## Example Test vs. Core Test

If you're not sure whether a test belongs in the examples directory or the main tests directory, consider:

- Put it in **examples/** if it primarily tests a user-facing example
- Put it in the main **tests/** directory if it tests core functionality
