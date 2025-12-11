"""
Calculator Tool

Provides basic mathematical operations for agents.
Uses AST-based evaluation to prevent malicious code execution.
"""

import ast
import math
import operator
from typing import Any, List
from .base import Tool, ToolParameter, ToolError
import re


# Safe operators for AST evaluation
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe functions for calculator
_SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}


def _safe_eval_ast(node: ast.AST) -> float:
    """
    Safely evaluate an AST node.
    
    Only allows:
    - Numbers (int, float)
    - Binary operations (+, -, *, /, //, %, **)
    - Unary operations (-, +)
    - Whitelisted function calls (sqrt, abs, round, etc.)
    
    Raises ValueError for any other node type.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)
    
    elif isinstance(node, ast.Constant):
        # Python 3.8+ uses ast.Constant for numbers
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    
    elif isinstance(node, ast.Num):
        # Legacy support for older Python versions
        return node.n
    
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        return _SAFE_OPERATORS[op_type](left, right)
    
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_ast(node.operand)
        return _SAFE_OPERATORS[op_type](operand)
    
    elif isinstance(node, ast.Call):
        # Only allow whitelisted function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in _SAFE_FUNCTIONS:
                raise ValueError(f"Unsupported function: {func_name}")
            func = _SAFE_FUNCTIONS[func_name]
            args = [_safe_eval_ast(arg) for arg in node.args]
            return func(*args)
        raise ValueError("Function calls must use simple names")
    
    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


class CalculatorTool(Tool):
    """
    A safe calculator tool for basic math operations.
    
    Supports: +, -, *, /, //, **, %, sqrt, abs, round, ceil, floor, sin, cos, tan, log, log10, exp
    Uses AST-based evaluation to prevent code injection.
    """
    
    def get_name(self) -> str:
        return "calculator"
    
    def get_description(self) -> str:
        return "Perform mathematical calculations. Supports +, -, *, /, //, ** (power), % (modulo), sqrt, abs, round, ceil, floor, sin, cos, tan, log, log10, exp."
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', '10 ** 2')",
                required=True
            )
        ]
    
    def is_safe(self, **kwargs) -> bool:
        """
        Check if the expression is safe to evaluate.
        
        This is a preliminary check. The actual AST-based evaluation
        provides the real safety guarantees.
        """
        expression = kwargs.get("expression", "")
        
        # Block obvious dangerous keywords as early check
        dangerous_keywords = [
            "import", "eval", "exec", "compile", "__",
            "open", "file", "input", "raw_input", "os.",
            "sys.", "subprocess", "lambda", "def ", "class "
        ]
        
        expr_lower = expression.lower()
        for keyword in dangerous_keywords:
            if keyword in expr_lower:
                return False
        
        return True
    
    async def execute(self, expression: str) -> Any:
        """
        Execute the calculation safely using AST-based evaluation.
        
        This approach parses the expression into an AST and only
        evaluates nodes that are explicitly whitelisted, preventing
        any code injection attacks.
        """
        try:
            # Parse the expression into an AST
            tree = ast.parse(expression, mode='eval')
            
            # Safely evaluate the AST
            result = _safe_eval_ast(tree)
            
            return {
                "expression": expression,
                "result": result
            }
            
        except ZeroDivisionError:
            raise ToolError("Division by zero", self.name)
        except SyntaxError as e:
            raise ToolError(f"Invalid expression syntax: {str(e)}", self.name)
        except ValueError as e:
            raise ToolError(f"Unsupported operation: {str(e)}", self.name)
        except Exception as e:
            raise ToolError(f"Calculation error: {str(e)}", self.name)
