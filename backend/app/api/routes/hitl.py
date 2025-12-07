"""
Human-in-the-Loop (HITL) API Routes

This module provides endpoints for managing human interventions in
agent execution workflows.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from backend.app.core.hitl import InterruptStatus, InterruptType


router = APIRouter(
    prefix="/hitl",
    tags=["human-in-the-loop"]
)


class InterruptResponse(BaseModel):
    """Response from human for an interrupt."""
    status: InterruptStatus
    response: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class InterruptInfo(BaseModel):
    """Information about an interrupt."""
    id: str
    type: InterruptType
    status: InterruptStatus
    node_name: str
    message: str
    data: Dict[str, Any]
    created_at: str
    responded_at: Optional[str] = None


class ProcessInterruptsResponse(BaseModel):
    """Response containing process interrupts."""
    process_id: str
    is_suspended: bool
    current_interrupt: Optional[InterruptInfo] = None
    pending_interrupts: list[InterruptInfo]


@router.get("/process/{process_id}/interrupts", response_model=ProcessInterruptsResponse)
async def get_process_interrupts(process_id: str):
    """
    Get all interrupts for a process.
    
    Returns information about pending and completed interrupts,
    including the currently active interrupt if the process is suspended.
    
    Args:
        process_id: The process identifier
        
    Returns:
        ProcessInterruptsResponse with interrupt information
        
    Example:
        ```
        GET /hitl/process/abc123/interrupts
        
        Response:
        {
            "process_id": "abc123",
            "is_suspended": true,
            "current_interrupt": {
                "id": "int-001",
                "type": "approval_required",
                "status": "pending",
                "node_name": "Drafting",
                "message": "Please review the draft decision",
                "data": {"draft": "..."},
                "created_at": "2025-12-07T10:00:00Z"
            },
            "pending_interrupts": [...]
        }
        ```
    """
    # TODO: Implement actual process state lookup from Redis or persistence layer
    # For now, return a placeholder response
    
    raise HTTPException(
        status_code=501,
        detail="HITL functionality requires integration with process management. "
               "This endpoint is a placeholder for the HITL architecture."
    )


@router.post("/process/{process_id}/interrupts/{interrupt_id}/respond")
async def respond_to_interrupt(
    process_id: str,
    interrupt_id: str,
    response: InterruptResponse
):
    """
    Respond to an interrupt and resume process execution.
    
    This endpoint allows a human to approve, reject, or modify an agent's
    proposed action, then resume the workflow.
    
    Args:
        process_id: The process identifier
        interrupt_id: The interrupt identifier
        response: Human's response to the interrupt
        
    Returns:
        Status of the response
        
    Example:
        ```
        POST /hitl/process/abc123/interrupts/int-001/respond
        {
            "status": "approved",
            "response": "Looks good, proceed",
            "data": {"confidence": "high"}
        }
        
        Response:
        {
            "success": true,
            "message": "Process resumed",
            "process_id": "abc123"
        }
        ```
    """
    # TODO: Implement actual interrupt response handling
    # This would:
    # 1. Load the process state from storage
    # 2. Update the interrupt with the human response
    # 3. Resume graph execution if appropriate
    # 4. Save the updated state
    
    raise HTTPException(
        status_code=501,
        detail="HITL functionality requires integration with process management. "
               "This endpoint is a placeholder for the HITL architecture."
    )


@router.get("/health")
async def hitl_health():
    """
    Health check endpoint for HITL service.
    
    Returns:
        Status message
    """
    return {
        "status": "healthy",
        "service": "human-in-the-loop",
        "note": "Full HITL integration pending"
    }
