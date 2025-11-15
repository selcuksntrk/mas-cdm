"""
Request Models - API request schemas

These models define the structure of incoming API requests
and provide automatic validation using Pydantic.

"""

from pydantic import BaseModel, Field, ConfigDict


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