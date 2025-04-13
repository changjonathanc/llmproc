"""Tests for the calculator tool."""

import math

import pytest

from llmproc.common.results import ToolResult
from llmproc.tools.builtin.calculator import calculator, safe_eval


@pytest.mark.asyncio
async def test_calculator_tool_basic_arithmetic():
    """Test basic arithmetic operations using the calculator tool."""
    # Addition
    result = await calculator("2 + 3")
    assert isinstance(result, str) or isinstance(result, ToolResult)
    if isinstance(result, ToolResult):
        assert result.content == "5"
    else:
        assert result == "5"

    # Subtraction
    result = await calculator("10 - 4")
    if isinstance(result, ToolResult):
        assert result.content == "6"
    else:
        assert result == "6"

    # Multiplication
    result = await calculator("6 * 7")
    if isinstance(result, ToolResult):
        assert result.content == "42"
    else:
        assert result == "42"

    # Division
    result = await calculator("10 / 2")
    if isinstance(result, ToolResult):
        assert result.content == "5"
    else:
        assert result == "5"

    # Integer division
    result = await calculator("10 // 3")
    if isinstance(result, ToolResult):
        assert result.content == "3"
    else:
        assert result == "3"

    # Modulo
    result = await calculator("10 % 3")
    if isinstance(result, ToolResult):
        assert result.content == "1"
    else:
        assert result == "1"

    # Exponentiation
    result = await calculator("2 ** 3")
    if isinstance(result, ToolResult):
        assert result.content == "8"
    else:
        assert result == "8"


@pytest.mark.asyncio
async def test_calculator_tool_complex_expressions():
    """Test more complex expressions with parentheses and multiple operations."""
    result = await calculator("2 * (3 + 4)")
    if isinstance(result, ToolResult):
        assert result.content == "14"
    else:
        assert result == "14"

    result = await calculator("(10 - 5) * (2 + 3)")
    if isinstance(result, ToolResult):
        assert result.content == "25"
    else:
        assert result == "25"

    result = await calculator("10 - 2 * 3")
    if isinstance(result, ToolResult):
        assert result.content == "4"
    else:
        assert result == "4"

    result = await calculator("(10 - 2) * 3")
    if isinstance(result, ToolResult):
        assert result.content == "24"
    else:
        assert result == "24"


@pytest.mark.asyncio
async def test_calculator_tool_mathematical_functions():
    """Test mathematical functions in the calculator tool."""
    # Square root
    result = await calculator("sqrt(16)")
    if isinstance(result, ToolResult):
        assert result.content == "4"
    else:
        assert result == "4"

    # Sine function
    result = await calculator("sin(0)")
    if isinstance(result, ToolResult):
        assert result.content == "0"
    else:
        assert result == "0"

    # Cosine function
    result = await calculator("cos(0)")
    if isinstance(result, ToolResult):
        assert result.content == "1"
    else:
        assert result == "1"

    # Absolute value
    result = await calculator("abs(-5)")
    if isinstance(result, ToolResult):
        assert result.content == "5"
    else:
        assert result == "5"

    # Logarithm
    result = await calculator("log10(100)")
    if isinstance(result, ToolResult):
        assert result.content == "2"
    else:
        assert result == "2"


@pytest.mark.asyncio
async def test_calculator_tool_constants():
    """Test mathematical constants in the calculator tool."""
    # Pi
    result = await calculator("pi")
    if isinstance(result, ToolResult):
        assert float(result.content) == pytest.approx(math.pi)
    else:
        assert float(result) == pytest.approx(math.pi)

    # e (Euler's number)
    result = await calculator("e")
    if isinstance(result, ToolResult):
        assert float(result.content) == pytest.approx(math.e)
    else:
        assert float(result) == pytest.approx(math.e)

    # Using constants in expressions
    result = await calculator("sin(pi/2)")
    if isinstance(result, ToolResult):
        assert float(result.content) == pytest.approx(1.0)
    else:
        assert float(result) == pytest.approx(1.0)

    result = await calculator("log(e)")
    if isinstance(result, ToolResult):
        assert float(result.content) == pytest.approx(1.0)
    else:
        assert float(result) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_calculator_tool_precision():
    """Test the precision parameter of the calculator tool."""
    # Default precision (6)
    result = await calculator("1/3")
    if isinstance(result, ToolResult):
        assert result.content == "0.333333"
    else:
        assert result == "0.333333"

    # Custom precision
    result = await calculator("1/3", 3)
    if isinstance(result, ToolResult):
        assert result.content == "0.333"
    else:
        assert result == "0.333"

    result = await calculator("1/3", 10)
    if isinstance(result, ToolResult):
        assert result.content == "0.3333333333"
    else:
        assert result == "0.3333333333"

    # Zero precision
    result = await calculator("pi", 0)
    if isinstance(result, ToolResult):
        assert result.content == "3"
    else:
        assert result == "3"

    # Invalid precision
    result = await calculator("1/3", -1)
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "Precision must be between" in result.content


@pytest.mark.asyncio
async def test_calculator_tool_error_handling():
    """Test error handling in the calculator tool."""
    # Division by zero
    result = await calculator("1/0")
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "division by zero" in result.content.lower()

    # Invalid expression
    result = await calculator("2 +* 3")
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "syntax error" in result.content.lower()

    # Undefined variable
    result = await calculator("x + 5")
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "unknown variable" in result.content.lower()

    # Invalid function call
    result = await calculator("sqrt(-1)")
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert (
        "math domain error" in result.content.lower()
        or "cannot convert" in result.content.lower()
    )

    # Function with wrong number of arguments
    result = await calculator("sin()")
    assert isinstance(result, ToolResult)
    assert result.is_error

    # Missing required parameter
    result = await calculator("")
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "must be a non-empty string" in result.content.lower()


@pytest.mark.asyncio
async def test_calculator_tool_security():
    """Test that the calculator tool properly restricts unsafe operations."""
    # Attempt to use built-in functions
    result = await calculator("__import__('os').system('ls')")
    assert isinstance(result, ToolResult)
    assert result.is_error

    # Attempt to use attribute access
    result = await calculator("''.join(['h', 'i'])")
    assert isinstance(result, ToolResult)
    assert result.is_error

    # Attempt to use list comprehension
    result = await calculator("[x for x in range(5)]")
    assert isinstance(result, ToolResult)
    assert result.is_error


def test_safe_eval_direct():
    """Test the safe_eval function directly for coverage."""
    # Basic operations
    assert safe_eval("2 + 3") == 5
    assert safe_eval("10 - 4") == 6
    assert safe_eval("6 * 7") == 42

    # Math functions
    assert safe_eval("sin(0)") == 0
    assert safe_eval("cos(0)") == 1
    assert safe_eval("sqrt(16)") == 4

    # Constants
    assert safe_eval("pi") == math.pi
    assert safe_eval("e") == math.e

    # Complex expressions
    assert safe_eval("2 * (3 + 4)") == 14
    assert safe_eval("sin(pi/2)") == 1.0

    # Test error cases
    with pytest.raises(ValueError):
        safe_eval("x + 5")  # Unknown variable

    with pytest.raises(ValueError):
        safe_eval("print('hello')")  # Disallowed function
