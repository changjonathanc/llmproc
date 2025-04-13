"""Tests for function-based implementation of existing tools."""

import math

import pytest

from llmproc.common.results import ToolResult
from llmproc.tools.builtin import calculator, read_file


@pytest.mark.asyncio
async def test_function_based_calculator():
    """Test the function-based calculator tool."""
    # Test basic arithmetic
    result = await calculator("2 + 3")
    assert result == "5"

    # Test complex expressions
    result = await calculator("2 * (3 + 4)")
    assert result == "14"

    # Test mathematical functions
    result = await calculator("sqrt(16) + 5")
    assert result == "9"

    result = await calculator("sin(pi/2)")
    assert float(result) == pytest.approx(1.0)

    # Test precision parameter
    result = await calculator("1/3", 3)
    assert result == "0.333"

    # Test error handling
    error_result = await calculator("1/0")
    assert isinstance(error_result, ToolResult)
    assert error_result.is_error is True
    assert "division by zero" in error_result.content.lower()


@pytest.mark.asyncio
async def test_function_based_read_file(tmp_path):
    """Test the function-based read_file tool."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")

    # Test reading the file
    result = await read_file(str(test_file))
    assert result == "Test content"

    # Test reading a non-existent file
    error_result = await read_file(str(tmp_path / "nonexistent.txt"))
    assert isinstance(error_result, ToolResult)
    assert error_result.is_error is True
    assert "not found" in error_result.content
