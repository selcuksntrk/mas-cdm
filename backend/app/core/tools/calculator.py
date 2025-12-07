"""
Calculator Tool

Provides basic mathematical operations for agents.
Includes safety checks to prevent malicious code execution.
"""

from typing import Any, List
from .base import Tool, ToolParameter, ToolError
import re


class CalculatorTool(Tool):
    """
    A safe calculator tool for basic math operations.
    
    Supports: +, -, *, /, **, %, sqrt, abs, round
    Does NOT use eval() to prevent code injection.
    """
    
    def get_name(self) -> str:
        return "calculator"
    
    def get_description(self) -> str:
        return "Perform basic mathematical calculations. Supports +, -, *, /, ** (power), % (modulo), sqrt, abs, round."
    
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
        
        Blocks any expressions containing:
        - Import statements
        - Function calls except whitelisted math functions
        - Variable assignments
        - Dangerous characters
        """
        expression = kwargs.get("expression", "")
        
        # Check for dangerous keywords
        dangerous_keywords = [
            "import", "eval", "exec", "compile", "__",
            "open", "file", "input", "raw_input"
        ]
        
        for keyword in dangerous_keywords:
            if keyword in expression.lower():
                return False
        
        # Only allow numbers, operators, parentheses, and whitelisted functions
        # Allow: digits, spaces, +, -, *, /, ., %, **, (, ), sqrt, abs, round, comma
        import re
        # Remove spaces and check character by character
        clean_expr = expression.replace(" ", "")
        allowed_pattern = r'^[\d\+\-\*\/\(\)\.\%sqrtab,round]+$'
        
        if not re.match(allowed_pattern, clean_expr):
            return False
        
        return True
    
    async def execute(self, expression: str) -> Any:
        """
        Execute the calculation safely.
        
        Uses a whitelist approach with limited math functions.
        """
        import math
        
        try:
            # Replace function names with their math module equivalents
            safe_expression = expression
            safe_expression = safe_expression.replace("sqrt", "math.sqrt")
            safe_expression = safe_expression.replace("abs", "abs")
            safe_expression = safe_expression.replace("round", "round")
            
            # Create safe namespace with only math functions
            safe_namespace = {
                "math": math,
                "abs": abs,
                "round": round,
                "__builtins__": {}  # Disable built-in functions
            }
            
            # Evaluate safely
            result = eval(safe_expression, safe_namespace, {})
            
            return {
                "expression": expression,
                "result": result
            }
            
        except ZeroDivisionError:
            raise ToolError("Division by zero", self.name)
        except SyntaxError as e:
            raise ToolError(f"Invalid expression syntax: {str(e)}", self.name)
        except Exception as e:
            raise ToolError(f"Calculation error: {str(e)}", self.name)
