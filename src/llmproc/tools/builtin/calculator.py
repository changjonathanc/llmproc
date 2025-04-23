"""Calculator tool for evaluating mathematical expressions."""

import ast
import math
import operator

from llmproc.common.access_control import AccessLevel
from llmproc.common.results import ToolResult
from llmproc.tools.function_tools import register_tool

# Allowed binary operators
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitXor: operator.xor,  # Used for power in some notations (^)
}

# Allowed mathematical functions
ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    # Trigonometric functions
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    # Other math functions
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "degrees": math.degrees,
    "radians": math.radians,
    "ceil": math.ceil,
    "floor": math.floor,
    "trunc": math.trunc,
    "factorial": math.factorial,
    "gcd": math.gcd,
}

# Mathematical constants
ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
}


class MathNode(ast.NodeVisitor):
    """AST node visitor for safely evaluating mathematical expressions."""

    def __init__(self):
        """Initialize the math evaluator."""
        self.constants = ALLOWED_CONSTANTS.copy()

    def visit_Constant(self, node):  # noqa: N802
        """Process constant values (numbers, etc.)."""
        return node.value

    def visit_Name(self, node):  # noqa: N802
        """Process variable names (constants like pi, e)."""
        if node.id in self.constants:
            return self.constants[node.id]
        raise ValueError(f"Unknown variable: {node.id}")

    def visit_BinOp(self, node):  # noqa: N802
        """Process binary operations (+, -, *, /, etc.)."""
        left = self.visit(node.left)
        right = self.visit(node.right)

        if type(node.op) not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported operation: {type(node.op).__name__}")

        return ALLOWED_OPERATORS[type(node.op)](left, right)

    def visit_UnaryOp(self, node):  # noqa: N802
        """Process unary operations (+x, -x)."""
        operand = self.visit(node.operand)

        if isinstance(node.op, ast.UAdd):
            return operand
        elif isinstance(node.op, ast.USub):
            return -operand
        else:
            raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")

    def visit_Call(self, node):  # noqa: N802
        """Process function calls (sin, cos, sqrt, etc.)."""
        func_name = getattr(node.func, "id", "")

        if func_name not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Function not allowed: {func_name}")

        args = [self.visit(arg) for arg in node.args]

        try:
            return ALLOWED_FUNCTIONS[func_name](*args)
        except Exception as e:
            raise ValueError(f"Error in function {func_name}: {str(e)}")

    def generic_visit(self, node):
        """Restrict any unhandled node types."""
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: A string containing a mathematical expression

    Returns:
        The calculated value as a float

    Raises:
        ValueError: If the expression is invalid or contains disallowed operations
    """
    # Parse the expression into an AST
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in expression: {str(e)}")

    # Evaluate the AST
    evaluator = MathNode()

    try:
        return evaluator.visit(tree.body)
    except ValueError as e:
        raise e
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")


@register_tool(
    description="""A tool for evaluating mathematical expressions safely.
It supports basic arithmetic, mathematical functions, and constants.""",
    param_descriptions={
        "expression": "The mathematical expression to evaluate. Supports operations like +, -, *, /, //, %, **, functions like sin, cos, sqrt, and constants like pi, e.",
        "precision": "Number of decimal places in the result (between 0 and 15, default: 6).",
    },
    # Calculator is marked as READ level because it's a pure function with no side effects:
    # - Does not modify any state
    # - Always returns the same output for the same input
    # - Does not access external resources
    access=AccessLevel.READ,
)
async def calculator(expression: str, precision: int = 6) -> str:
    """Calculate the result of a mathematical expression.

    Args:
        expression: The mathematical expression to evaluate
        precision: Number of decimal places in the result (default: 6)

    Returns:
        The calculated result as a string
    """
    # Validate parameters
    if not expression or not isinstance(expression, str):
        return ToolResult.from_error("Expression must be a non-empty string.")

    try:
        precision = int(precision)
        if precision < 0 or precision > 15:
            return ToolResult.from_error("Precision must be between 0 and 15.")
    except ValueError:
        return ToolResult.from_error("Precision must be an integer.")

    # Try to evaluate the expression
    try:
        result = safe_eval(expression)

        # Format result based on type and precision
        if isinstance(result, int | float):
            if math.isnan(result):
                formatted_result = "NaN"
            elif math.isinf(result):
                formatted_result = "Infinity" if result > 0 else "-Infinity"
            else:
                # Apply precision
                formatted_result = f"{result:.{precision}f}".rstrip("0").rstrip(".")
                # Convert to int if it's a whole number
                if formatted_result.endswith(".0"):
                    formatted_result = formatted_result[:-2]
        else:
            formatted_result = str(result)

        return formatted_result

    except ValueError as e:
        return ToolResult.from_error(f"Calculation error: {str(e)}")
    except Exception as e:
        return ToolResult.from_error(f"Unexpected error: {str(e)}")
