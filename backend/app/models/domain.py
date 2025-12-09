"""
Domain Models - Core business entities

These represent the core concepts in your application domain.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict




# Decision State Model
class DecisionState(BaseModel):
    """
    State object that tracks the decision-making process
    
    This is passed through the entire graph execution
    and accumulates information at each step.
    
    WHY PYDANTIC INSTEAD OF DATACLASS:
    - Automatic validation
    - JSON serialization/deserialization
    - Better integration with FastAPI and Pydantic AI
    - Type coercion (strings to ints, etc.)
    - Immutability options (frozen=True)
    """
    
    model_config = ConfigDict(
        # Allow arbitrary types (for complex objects)
        arbitrary_types_allowed=True,
        # Validate on assignment (catch errors early)
        validate_assignment=True,
        # Use enum values instead of enum objects in JSON
        use_enum_values=True,
    )
    
    # Metadata
    decision_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this decision process (for tracing and correlation)"
    )
    
    # User Input
    decision_requested: str = Field(
        default="",
        description="The original decision query from the user"
    )
    
    # Analysis Phase
    trigger: str = Field(
        default="",
        description="Identified trigger for the decision (opportunity, problem, crisis)"
    )
    
    root_cause: str = Field(
        default="",
        description="Root cause analysis using 5 Whys or Fishbone"
    )
    
    scope_definition: str = Field(
        default="",
        description="Defined scope of the decision (what's in/out of scope)"
    )
    
    # Drafting Phase
    decision_drafted: str = Field(
        default="",
        description="Initial drafted decision document"
    )
    
    goals: str = Field(
        default="",
        description="Established goals and success metrics (SMART format)"
    )
    
    stakeholders: str = Field(
        default="",
        description="Identified stakeholders and their interests"
    )
    
    # Information Gathering
    complementary_info: str = Field(
        default="",
        description="Additional information gathered to inform decision"
    )
    
    complementary_info_num: int = Field(
        default=0,
        ge=0,  # Greater than or equal to 0
        description="Number of information items gathered"
    )

    # Memory / Retrieval
    memory_context: str = Field(
        default="",
        description="Retrieved context from memory store"
    )
    memory_hits: int = Field(
        default=0,
        ge=0,
        description="Number of memory documents retrieved for the decision"
    )
    
    # Refinement Phase
    decision_draft_updated: str = Field(
        default="",
        description="Updated decision draft after information gathering"
    )
    
    generated_alternatives: str = Field(
        default="",
        description="Generated alternative options"
    )
    
    alternatives: str = Field(
        default="",
        description="Formatted alternatives with evaluation criteria"
    )
    
    # Final Results
    result: str = Field(
        default="",
        description="The selected decision option"
    )
    
    result_comment: str = Field(
        default="",
        description="Explanation of why this option was selected"
    )
    
    best_alternative_result: str = Field(
        default="",
        description="The best alternative option (runner-up)"
    )
    
    best_alternative_result_comment: str = Field(
        default="",
        description="Explanation of the alternative option"
    )
    

# Result Output Model
class ResultOutput(BaseModel):
    """
    Output structure for the final decision result.
    Used by the result agent to return structured decision information.

    The example in `model_config` is used for OpenAPI schema generation and documentation.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "result": "Implement Redis persistence with Repository Pattern",
                "best_alternative_result": "Use PostgreSQL with SQLAlchemy ORM",
                "result_comment": "Redis provides faster access and simpler setup for our use case",
                "best_alternative_result_comment": "PostgreSQL would offer more complex querying but is overkill"
            }
        }
    )
    
    result: str = Field(
        ...,
        min_length=1,
        description="The selected option for the decision"
    )
    
    best_alternative_result: str = Field(
        ...,
        min_length=1,
        description="The best alternative option for the decision"
    )
    
    result_comment: str = Field(
        ...,
        min_length=1,
        description="Comment on the selection of the result"
    )
    
    best_alternative_result_comment: str = Field(
        ...,
        min_length=1,
        description="Comment on the selection of the best alternative to result"
    )
    

# Evaluation Output Model
class EvaluationOutput(BaseModel):
    """
    Output structure for agent evaluations.
    Used by evaluator agents to provide feedback on agent outputs.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "correct": True,
                "comment": "The root cause analysis properly identified the underlying issue using 5 Whys technique"
            }
        }
    )
    
    correct: bool = Field(
        ...,
        description="Whether the answer is correct"
    )
    
    comment: str = Field(
        ...,
        min_length=5,
        description="Comment on the answer with specific feedback (minimum 5 characters)"
    )
    
    
# Process Info Model
class ProcessInfo(BaseModel):
    """
    Information about a running decision-making process
    
    WHY PYDANTIC INSTEAD OF DATACLASS:
    - Validation when loading from Redis
    - Consistent serialization format
    - Integration with repository pattern
    """
    
    model_config = ConfigDict(
        # No unsupported config options
    )
    
    process_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the process (UUID string)"
    )
    
    # The original decision query submitted by the user
    query: str = Field(
        default="",
        description="The original decision query for this process"
    )
    
    status: str = Field(
        ...,
        pattern="^(pending|running|completed|failed)$",
        description="Process status: pending, running, completed, or failed"
    )
    
    # NEW: Track current step for real-time progress
    current_step: Optional[str] = Field(
        default=None,
        description="Current step being executed (for progress tracking)"
    )
    
    # NEW: Track completed steps
    completed_steps: list[str] = Field(
        default_factory=list,
        description="List of completed step names"
    )
    
    result: Optional[DecisionState] = Field(
        default=None,
        description="The complete decision state (only when completed)"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if process failed"
    )
    
    # Accept either an ISO formatted string or a datetime to make tests and
    # repository implementations easier (they sometimes pass datetime objects).
    created_at: Optional[datetime | str] = Field(
        default=None,
        description="Timestamp when process was created (datetime or ISO string)"
    )

    completed_at: Optional[datetime | str] = Field(
        default=None,
        description="Timestamp when process completed (datetime or ISO string)"
    )