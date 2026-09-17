"""
Calculator Tool — safe AST-based mathematical expression evaluator.

Security design:
  - Uses Python's ast module to parse expressions into a syntax tree.
  - Only whitelisted node types (constants, binary ops, unary ops, safe function calls) are allowed.
  - eval() and exec() are NEVER used.
  - Any attempt to call unsafe functions or access attributes raises ValueError.
"""
from __future__ import annotations

import ast
import math
import operator
import time
from typing import Any

from backend.tools.base import BaseTool, ToolResult

# ── Whitelist of safe AST node types → Python operators ──────────────────────

_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES: dict[str, Any] = {
    # Math functions
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "hypot": math.hypot,
    # Constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


class SafeEvaluator:
    """
    Walks a Python AST and evaluates only whitelisted operations.
    Raises ValueError for any unsafe or unrecognized node.
    """

    def evaluate(self, expression: str) -> float | int:
        """Parse and evaluate a mathematical expression string safely."""
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression syntax: {exc}") from exc
        return self._eval(tree.body)

    def _eval(self, node: ast.AST) -> float | int:
        # Numeric literal: 42, 3.14, etc.
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

        # Named constant or function: pi, sqrt, etc.
        if isinstance(node, ast.Name):
            if node.id in _SAFE_NAMES:
                return _SAFE_NAMES[node.id]  # type: ignore[return-value]
            raise ValueError(f"Unknown name '{node.id}'. Allowed: {list(_SAFE_NAMES)}")

        # Binary operation: a + b, a * b, etc.
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BINARY_OPS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            left = self._eval(node.left)
            right = self._eval(node.right)
            try:
                return _BINARY_OPS[op_type](left, right)
            except ZeroDivisionError:
                raise ValueError("Division by zero.")

        # Unary operation: -x, +x
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._eval(node.operand)
            return _UNARY_OPS[op_type](operand)

        # Function call: sqrt(4), round(3.7), etc.
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed (no method calls).")
            func_name = node.func.id
            if func_name not in _SAFE_NAMES:
                raise ValueError(f"Function '{func_name}' is not allowed.")
            func = _SAFE_NAMES[func_name]
            if not callable(func):
                raise ValueError(f"'{func_name}' is a constant, not a function.")
            args = [self._eval(a) for a in node.args]
            return func(*args)

        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


class CalculatorTool(BaseTool):
    """
    Safe mathematical expression evaluator.

    Supports: arithmetic, exponentiation, trigonometry, logarithms,
    rounding, abs, factorial, and math constants (pi, e, tau).
    Never uses eval() or exec().
    """

    name = "calculator"
    description = (
        "Evaluates mathematical and arithmetic expressions safely. "
        "Supports: +, -, *, /, //, %, ** (power), sqrt, log, log10, log2, exp, "
        "sin, cos, tan, asin, acos, atan, abs, round, ceil, floor, factorial, "
        "hypot, gcd. Constants: pi, e, tau. "
        "Input must be a valid mathematical expression string like '25 * 47 + 100' "
        "or '10000 * (1 + 0.05) ** 3'."
    )
    capabilities = [
        "arithmetic calculations",
        "mathematical expressions",
        "compound interest formula",
        "percentage calculations",
        "trigonometric calculations",
        "logarithmic calculations",
        "scientific calculations",
        "unit conversion via formula",
    ]
    input_schema = {
        "expression": {
            "type": "string",
            "description": (
                "Mathematical expression to evaluate. "
                "Example: '25 * 47 + 100' or '10000 * (1.05 ** 3)'"
            ),
            "required": True,
        }
    }

    def __init__(self) -> None:
        self._evaluator = SafeEvaluator()

    def execute(self, input_data: dict[str, Any]) -> ToolResult:
        start = time.monotonic()
        goal_id = input_data.get("goal_id", "unknown")
        expression = str(input_data.get("expression", "")).strip()

        if not expression:
            return ToolResult(
                success=False,
                output=None,
                error="No expression provided. Supply an 'expression' key.",
                execution_time=time.monotonic() - start,
                tool_name=self.name,
                goal_id=goal_id,
            )

        try:
            result = self._evaluator.evaluate(expression)
            # Format nicely: int if result is whole number
            formatted = int(result) if isinstance(result, float) and result.is_integer() else result
            return ToolResult(
                success=True,
                output={
                    "expression": expression,
                    "result": formatted,
                    "result_str": f"{expression} = {formatted}",
                },
                error=None,
                execution_time=time.monotonic() - start,
                tool_name=self.name,
                goal_id=goal_id,
            )
        except (ValueError, TypeError, OverflowError) as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Cannot evaluate '{expression}': {exc}",
                execution_time=time.monotonic() - start,
                tool_name=self.name,
                goal_id=goal_id,
            )
