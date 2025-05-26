"""Standard test patterns and helper functions for LLMProc tests.

This module provides standardized patterns and utilities for writing tests.
It reduces duplication across test files and encourages consistent test organization.

# Test Categories
- Unit Tests: Test individual functions in isolation with mocks
- Integration Tests: Test interactions between components without API calls
- API Tests: Test functionality with real LLM providers
- Configuration Tests: Test loading and validating configurations

# Key Patterns
- Arrange-Act-Assert structure: Clear separation of setup, action, and verification
- Fixture-based setup: Use fixtures for common setup rather than inline setup
- Dependency injection: Pass dependencies as arguments rather than creating them inline
- Isolation: Each test should run independently of others
- Clear naming: Test names should describe what they're testing

Example Usage:
```python
import pytest
from tests.patterns import timed_test, assert_successful_response

@pytest.mark.llm_api
@pytest.mark.asyncio
async def test_feature(mocked_llm_process):
    # Arrange - using fixture

    # Act - with timing check
    with timed_test(timeout_seconds=3.0):
        result = await mocked_llm_process.run("Test prompt")

    # Assert - using helper
    assert_successful_response(result)
```
"""

import asyncio
import contextlib
import inspect
import time
from collections.abc import Callable
from typing import Any, Optional, Union

import pytest
from llmproc.common.results import RunResult, ToolResult


@contextlib.contextmanager
def timed_test(timeout_seconds: float = 8.0):
    """Provides timing context for API tests.

    Args:
        timeout_seconds: Maximum allowed duration in seconds

    Yields:
        None: Just provides the timing context

    Raises:
        AssertionError: If the test exceeds the timeout
    """
    start_time = time.time()
    yield
    duration = time.time() - start_time
    assert duration < timeout_seconds, f"Test took too long: {duration:.2f}s > {timeout_seconds}s timeout"


def assert_successful_response(result: Union[RunResult, ToolResult, str]):
    """Asserts that a response from an LLM or tool was successful.

    Args:
        result: The response to check (RunResult, ToolResult, or string)

    Raises:
        AssertionError: If the response indicates an error
    """
    if isinstance(result, ToolResult):
        assert not result.is_error, f"Expected successful result, but got error: {result.content}"
    elif isinstance(result, RunResult):
        assert (
            result.last_message.strip() if hasattr(result, "last_message") else True
        ), "Expected non-empty response content"
    elif isinstance(result, str):
        assert result.strip(), "Expected non-empty response content"
    else:
        pytest.fail(f"Unknown result type: {type(result)}")


def assert_error_response(result: Union[RunResult, ToolResult, str], error_text: str = None):
    """Asserts that a response from an LLM or tool was an error.

    Args:
        result: The response to check (RunResult, ToolResult, or string)
        error_text: Optional substring to look for in the error message

    Raises:
        AssertionError: If the response doesn't indicate an error or doesn't contain expected text
    """
    if isinstance(result, ToolResult):
        assert result.is_error, "Expected error result, but got success"
        if error_text:
            assert error_text in result.content, f"Expected error to contain '{error_text}'"
    elif isinstance(result, str):
        if error_text:
            assert error_text in result, f"Expected error to contain '{error_text}'"
    else:
        pytest.fail(f"Unknown result type: {type(result)}")


# Additional utility functions can be added here as the test suite evolves
