"""
Request Models - API request schemas

These models define the structure of incoming API requests
and provide automatic validation using Pydantic.

"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Tuple


# Decision Request Model
class DecisionRequest(BaseModel):
    """
    Request model for starting a decision-making process

    Example:
        {
            "decision_query": "Should I invest in AI startups?"
        }
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "decision_query": "Should I invest in AI startups?"
                },
                {
                    "decision_query": "Should I expand my business to international markets?"
                }
            ]
        }
    )
    
    decision_query: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The decision you need help with"
    )


class BuildGraphRequest(BaseModel):
    """Request model for building a dynamic graph."""

    nodes: List[str] = Field(
        ...,
        min_length=2,
        description="List of node class names to include (must contain GetDecision)",
    )
    edges: List[Tuple[str, str]] | None = Field(
        default=None,
        description="Optional list of directed edges (source, target). If omitted, nodes are wired linearly.",
    )


class AgentConfigUpdateRequest(BaseModel):
    """Request payload for updating per-agent configuration."""

    model: str | None = Field(
        default=None,
        description="Override model identifier for the agent",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Optional temperature override for the agent",
    )