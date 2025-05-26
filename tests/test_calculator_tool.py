"""Unit tests for the calculator tool.

This file follows the standardized unit test patterns:
1. Clear Arrange-Act-Assert structure
2. Focus on testing inputs and outputs of individual functions
3. Direct import of only what's needed for the test
4. Use of pytest.mark.parametrize for efficient testing of multiple inputs
"""

import math
from typing import Union

import pytest
from llmproc.common.results import ToolResult
from llmproc.tools.builtin.calculator import calculator, safe_eval


class TestSafeEval:
    """Unit tests for the safe_eval function."""

    @pytest.mark.parametrize(
        "expression,expected",
        [
            # Basic arithmetic
            ("2 + 3", 5),
            ("10 - 4", 6),
            ("6 * 7", 42),
            ("10 / 2", 5),
            ("10 // 3", 3),
            ("10 % 3", 1),
            ("2 ** 3", 8),
            # Complex expressions
            ("2 * (3 + 4)", 14),
            ("(10 - 5) * (2 + 3)", 25),
            ("10 - 2 * 3", 4),
            ("(10 - 2) * 3", 24),
            # Math functions
            ("sqrt(16)", 4),
            ("sin(0)", 0),
            ("cos(0)", 1),
            ("abs(-5)", 5),
            ("log10(100)", 2),
            # Constants
            ("pi", math.pi),
            ("e", math.e),
            # Complex with constants
            ("sin(pi/2)", 1.0),
            ("log(e)", 1.0),
        ],
    )
    def test_valid_expressions(self, expression: str, expected: float):
        """Test that safe_eval correctly evaluates valid mathematical expressions.

        Args:
            expression: The expression to evaluate
            expected: The expected result
        """
        # Act
        result = safe_eval(expression)

        # Assert
        if isinstance(expected, float):
            assert result == pytest.approx(expected)
        else:
            assert result == expected

    @pytest.mark.parametrize(
        "expression,error_type,error_message",
        [
            # Syntax errors
            ("2 +* 3", ValueError, "syntax error"),
            # Unknown variables
            ("x + 5", ValueError, "unknown variable"),
            # Disallowed functions
            ("__import__('os').system('ls')", ValueError, None),
            ("print('hello')", ValueError, None),
            # Security restrictions
            ("''.join(['h', 'i'])", ValueError, None),
            ("[x for x in range(5)]", ValueError, None),
        ],
    )
    def test_invalid_expressions(self, expression: str, error_type: type, error_message: str):
        """Test that safe_eval properly handles invalid or unsafe expressions.

        Args:
            expression: The expression to evaluate
            error_type: The expected exception type
            error_message: Expected text in the error message, or None if any error is fine
        """
        # Arrange & Act & Assert
        with pytest.raises(error_type) as excinfo:
            safe_eval(expression)

        # Optional more specific error message check
        if error_message:
            assert error_message in str(excinfo.value).lower()


class TestCalculatorTool:
    """Integration tests for the calculator tool function."""

    async def _assert_calculator_result(self, expression: str, expected: str, precision: int = None):
        """Helper to assert calculator tool results.

        Args:
            expression: Expression to calculate
            expected: Expected result string
            precision: Optional precision parameter
        """
        # Act
        if precision is not None:
            result = await calculator(expression, precision)
        else:
            result = await calculator(expression)

        # Assert - Handle both string and ToolResult return types
        if isinstance(result, ToolResult):
            assert result.content == expected
        else:
            assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "expression,expected",
        [
            # Basic arithmetic
            ("2 + 3", "5"),
            ("10 - 4", "6"),
            ("6 * 7", "42"),
            ("10 / 2", "5"),
            ("10 // 3", "3"),
            ("10 % 3", "1"),
            ("2 ** 3", "8"),
        ],
    )
    async def test_basic_arithmetic(self, expression: str, expected: str):
        """Test basic arithmetic operations using the calculator tool.

        Args:
            expression: The expression to evaluate
            expected: The expected result string
        """
        await self._assert_calculator_result(expression, expected)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("2 * (3 + 4)", "14"),
            ("(10 - 5) * (2 + 3)", "25"),
            ("10 - 2 * 3", "4"),
            ("(10 - 2) * 3", "24"),
        ],
    )
    async def test_complex_expressions(self, expression: str, expected: str):
        """Test more complex expressions with parentheses and multiple operations.

        Args:
            expression: The expression to evaluate
            expected: The expected result string
        """
        await self._assert_calculator_result(expression, expected)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("sqrt(16)", "4"),
            ("sin(0)", "0"),
            ("cos(0)", "1"),
            ("abs(-5)", "5"),
            ("log10(100)", "2"),
        ],
    )
    async def test_mathematical_functions(self, expression: str, expected: str):
        """Test mathematical functions in the calculator tool.

        Args:
            expression: The expression to evaluate
            expected: The expected result string
        """
        await self._assert_calculator_result(expression, expected)

    @pytest.mark.asyncio
    async def test_constants(self):
        """Test mathematical constants in the calculator tool."""
        # Arrange & Act & Assert
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
    @pytest.mark.parametrize(
        "expression,precision,expected",
        [
            ("1/3", None, "0.333333"),  # Default precision (6)
            ("1/3", 3, "0.333"),  # Custom precision
            ("1/3", 10, "0.3333333333"),  # Higher precision
            ("pi", 0, "3"),  # Zero precision
        ],
    )
    async def test_precision(self, expression: str, precision: int, expected: str):
        """Test the precision parameter of the calculator tool.

        Args:
            expression: The expression to evaluate
            precision: The precision parameter value
            expected: The expected result string
        """
        await self._assert_calculator_result(expression, expected, precision)

    @pytest.mark.asyncio
    async def test_precision_error(self):
        """Test error handling for invalid precision values."""
        # Arrange & Act
        result = await calculator("1/3", -1)

        # Assert
        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "Precision must be between" in result.content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "expression,error_text",
        [
            ("1/0", "division by zero"),
            ("2 +* 3", "syntax error"),
            ("x + 5", "unknown variable"),
            ("sqrt(-1)", None),  # Either "math domain error" or "cannot convert"
            ("sin()", None),  # Any error is fine
            ("", "must be a non-empty string"),
        ],
    )
    async def test_error_handling(self, expression: str, error_text: str):
        """Test error handling in the calculator tool.

        Args:
            expression: The expression to evaluate
            error_text: Expected text in the error message, or None if any error is fine
        """
        # Arrange & Act
        result = await calculator(expression)

        # Assert
        assert isinstance(result, ToolResult)
        assert result.is_error
        if error_text:
            assert error_text.lower() in result.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('ls')",
            "''.join(['h', 'i'])",
            "[x for x in range(5)]",
        ],
    )
    async def test_security_restrictions(self, expression: str):
        """Test that the calculator tool properly restricts unsafe operations.

        Args:
            expression: A potentially unsafe expression that should be rejected
        """
        # Arrange & Act
        result = await calculator(expression)

        # Assert
        assert isinstance(result, ToolResult)
        assert result.is_error
