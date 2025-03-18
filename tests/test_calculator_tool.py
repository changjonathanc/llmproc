"""Tests for the calculator tool."""

import math

import pytest

from llmproc.tools.calculator import calculator_tool, safe_eval
from llmproc.tools.tool_result import ToolResult


@pytest.mark.asyncio
async def test_calculator_tool_basic_arithmetic():
    """Test basic arithmetic operations using the calculator tool."""
    # Addition
    result = await calculator_tool("2 + 3")
    assert isinstance(result, ToolResult)
    assert result.content == "5"

    # Subtraction
    result = await calculator_tool("10 - 4")
    assert result.content == "6"

    # Multiplication
    result = await calculator_tool("6 * 7")
    assert result.content == "42"

    # Division
    result = await calculator_tool("10 / 2")
    assert result.content == "5"

    # Integer division
    result = await calculator_tool("10 // 3")
    assert result.content == "3"

    # Modulo
    result = await calculator_tool("10 % 3")
    assert result.content == "1"

    # Exponentiation
    result = await calculator_tool("2 ** 3")
    assert result.content == "8"


@pytest.mark.asyncio
async def test_calculator_tool_complex_expressions():
    """Test more complex expressions with parentheses and multiple operations."""
    result = await calculator_tool("2 * (3 + 4)")
    assert result.content == "14"

    result = await calculator_tool("(10 - 5) * (2 + 3)")
    assert result.content == "25"

    result = await calculator_tool("10 - 2 * 3")
    assert result.content == "4"

    result = await calculator_tool("(10 - 2) * 3")
    assert result.content == "24"


@pytest.mark.asyncio
async def test_calculator_tool_mathematical_functions():
    """Test mathematical functions in the calculator tool."""
    # Square root
    result = await calculator_tool("sqrt(16)")
    assert result.content == "4"

    # Sine function
    result = await calculator_tool("sin(0)")
    assert result.content == "0"

    # Cosine function
    result = await calculator_tool("cos(0)")
    assert result.content == "1"

    # Absolute value
    result = await calculator_tool("abs(-5)")
    assert result.content == "5"

    # Logarithm
    result = await calculator_tool("log10(100)")
    assert result.content == "2"


@pytest.mark.asyncio
async def test_calculator_tool_constants():
    """Test mathematical constants in the calculator tool."""
    # Pi
    result = await calculator_tool("pi")
    assert float(result.content) == pytest.approx(math.pi)

    # e (Euler's number)
    result = await calculator_tool("e")
    assert float(result.content) == pytest.approx(math.e)

    # Using constants in expressions
    result = await calculator_tool("sin(pi/2)")
    assert float(result.content) == pytest.approx(1.0)

    result = await calculator_tool("log(e)")
    assert float(result.content) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_calculator_tool_precision():
    """Test the precision parameter of the calculator tool."""
    # Default precision (6)
    result = await calculator_tool("1/3")
    assert result.content == "0.333333"

    # Custom precision
    result = await calculator_tool("1/3", precision=3)
    assert result.content == "0.333"

    result = await calculator_tool("1/3", precision=10)
    assert result.content == "0.3333333333"

    # Zero precision
    result = await calculator_tool("pi", precision=0)
    assert result.content == "3"

    # Invalid precision
    result = await calculator_tool("1/3", precision=-1)
    assert result.is_error
    assert "Precision must be between" in result.content


@pytest.mark.asyncio
async def test_calculator_tool_error_handling():
    """Test error handling in the calculator tool."""
    # Division by zero
    result = await calculator_tool("1/0")
    assert result.is_error
    assert "division by zero" in result.content.lower()

    # Invalid expression
    result = await calculator_tool("2 +* 3")
    assert result.is_error
    assert "syntax error" in result.content.lower()

    # Undefined variable
    result = await calculator_tool("x + 5")
    assert result.is_error
    assert "unknown variable" in result.content.lower()

    # Invalid function call
    result = await calculator_tool("sqrt(-1)")
    assert result.is_error
    assert (
        "math domain error" in result.content.lower()
        or "cannot convert" in result.content.lower()
    )

    # Function with wrong number of arguments
    result = await calculator_tool("sin()")
    assert result.is_error

    # Missing required parameter
    result = await calculator_tool("")
    assert result.is_error
    assert "must be a non-empty string" in result.content.lower()


@pytest.mark.asyncio
async def test_calculator_tool_security():
    """Test that the calculator tool properly restricts unsafe operations."""
    # Attempt to use built-in functions
    result = await calculator_tool("__import__('os').system('ls')")
    assert result.is_error

    # Attempt to use attribute access
    result = await calculator_tool("''.join(['h', 'i'])")
    assert result.is_error

    # Attempt to use list comprehension
    result = await calculator_tool("[x for x in range(5)]")
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
