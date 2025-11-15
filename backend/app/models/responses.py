"""
Response Models - API response schemas

These models define the structure of outgoing API responses.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# Decision Response Model
class DecisionResponse(BaseModel):
    """
    Complete decision-making result
    
    Contains the final decision, alternatives, and the reasoning process.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "selected_decision": "Yes, invest in renewable energy stocks with diversified portfolio",
                "selected_decision_comment": "Based on market trends and environmental goals...",
                "alternative_decision": "Consider green energy ETFs instead",
                "alternative_decision_comment": "Lower risk with similar environmental impact...",
                "trigger": "Market opportunity in renewable sector",
                "root_cause": "Global shift towards sustainability",
                "scope_definition": "Investment decision in renewable energy sector",
                "decision_drafted": "Initial investment recommendation...",
                "goals": "Maximize returns while supporting sustainability"
            }
        }
    )
    
    # Final Results
    selected_decision: str = Field(
        ...,
        description="The recommended decision"
    )
    
    selected_decision_comment: str = Field(
        ...,
        description="Explanation of why this decision was selected"
    )
    
    alternative_decision: str = Field(
        ...,
        description="The best alternative option"
    )
    
    alternative_decision_comment: str = Field(
        ...,
        description="Explanation of the alternative option"
    )
    
    # Process Information
    trigger: str = Field(
        default="",
        description="Identified trigger for the decision"
    )
    
    root_cause: str = Field(
        default="",
        description="Root cause analysis"
    )
    
    scope_definition: str = Field(
        default="",
        description="Defined scope of the decision"
    )
    
    decision_drafted: str = Field(
        default="",
        description="Initial drafted decision"
    )
    
    goals: str = Field(
        default="",
        description="Established goals for the decision"
    )
    
    complementary_info: str = Field(
        default="",
        description="Additional information gathered"
    )
    
    decision_draft_updated: str = Field(
        default="",
        description="Updated draft after information gathering"
    )
    
    alternatives: str = Field(
        default="",
        description="Generated alternatives"
    )
    
    
# Process Status Response Model
class ProcessStatusResponse(BaseModel):
    """Response for async process status check"""
    
    process_id: str = Field(..., description="Unique process identifier")
    status: str = Field(..., description="Process status: running, completed, failed")
    created_at: Optional[str] = Field(None, description="Process creation timestamp")
    completed_at: Optional[str] = Field(None, description="Process completion timestamp")
    current_step: Optional[str] = Field(None, description="Current step being executed")
    completed_steps: list[str] = Field(default_factory=list, description="List of completed steps")
    result: Optional[dict] = Field(None, description="Result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")


# Process Start Response Model
class ProcessStartResponse(BaseModel):
    """Response when starting an async process"""
    
    process_id: str = Field(..., description="Unique process identifier")
    status: str = Field(..., description="Process status")
    message: str = Field(..., description="Status message")
    
    
# Health Check Response Model
class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    message: str = Field(default="", description="Additional information")
    version: str = Field(default="", description="API version")
    endpoints: Optional[dict] = Field(None, description="Available API endpoints")
    

# Mermaid Diagram Response Model
class MermaidResponse(BaseModel):
    """Mermaid diagram response"""
    
    mermaid_code: str = Field(..., description="Mermaid diagram code for visualization")
    

# Error Response Model
class ErrorResponse(BaseModel):
    """Error response"""
    
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: Optional[str] = Field(None, description="Error timestamp")