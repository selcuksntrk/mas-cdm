"""
Helper Utilities

Common utility functions used across the application.
"""

from pathlib import Path
from typing import Optional
import logging


# Setup logger
logger = logging.getLogger(__name__)


def load_prompt(filename: str, prompts_dir: Optional[Path] = None) -> str:
    """
    Load a system prompt from a text file.
    
    Args:
        filename: Name of the prompt file (e.g., "identify_trigger_agent.txt")
        prompts_dir: Directory containing prompts (defaults to configured path)
        
    Returns:
        The prompt text content
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
        
    Example:
        >>> prompt = load_prompt("drafting_agent.txt")
    """
    if prompts_dir is None:
        # Default to templates directory relative to this file
        prompts_dir = Path(__file__).parent.parent / "core" / "prompts" / "templates"
    
    prompt_path = prompts_dir / filename
    
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}\n"
            f"Looking in: {prompts_dir}"
        )
    
    try:
        content = prompt_path.read_text(encoding="utf-8")
        logger.debug(f"Loaded prompt from {filename}")
        return content
    except Exception as e:
        logger.error(f"Error loading prompt {filename}: {e}")
        raise


def format_decision_context(context: dict) -> str:
    """
    Format decision context for agent prompts.
    
    Args:
        context: Dictionary of context information
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in context.items():
        if value:  # Only include non-empty values
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def validate_decision_query(query: str) -> tuple[bool, Optional[str]]:
    """
    Validate a decision query.
    
    Args:
        query: The decision query to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Decision query cannot be empty"
    
    if len(query) < 10:
        return False, "Decision query is too short (minimum 10 characters)"
    
    if len(query) > 1000:
        return False, "Decision query is too long (maximum 1000 characters)"
    
    return True, None
