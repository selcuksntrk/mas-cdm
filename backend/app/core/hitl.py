"""
Human-in-the-Loop (HITL) Support

This module provides functionality for suspending and resuming graph execution
to allow human intervention and approval at critical decision points.
"""

from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, UTC
from pydantic import BaseModel, Field


class InterruptType(str, Enum):
    """Types of interrupts that can occur during execution."""
    APPROVAL_REQUIRED = "approval_required"
    INPUT_REQUIRED = "input_required"
    REVIEW_REQUIRED = "review_required"
    FEEDBACK_REQUIRED = "feedback_required"


class InterruptStatus(str, Enum):
    """Status of an interrupt."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    CANCELLED = "cancelled"


class InterruptEvent(BaseModel):
    """
    Represents an interrupt event in the graph execution.
    
    When an agent needs human approval or input, it creates an interrupt event
    that pauses execution until a human responds.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: InterruptType
    status: InterruptStatus = InterruptStatus.PENDING
    
    # Context about the interrupt
    node_name: str = Field(description="Name of the node that triggered the interrupt")
    message: str = Field(description="Message explaining why human input is needed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Data associated with the interrupt")
    
    # Response from human
    human_response: Optional[str] = None
    human_data: Optional[Dict[str, Any]] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    responded_at: Optional[datetime] = None
    
    def respond(
        self,
        status: InterruptStatus,
        response: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Record human response to the interrupt.
        
        Args:
            status: New status (approved, rejected, modified)
            response: Human's response message
            data: Additional data from human
        """
        self.status = status
        self.human_response = response
        self.human_data = data or {}
        self.responded_at = datetime.now(UTC)


class HITLState(BaseModel):
    """
    State extension for Human-in-the-Loop functionality.
    
    This can be mixed into DecisionState to add HITL capabilities.
    """
    
    # Track active interrupts
    interrupts: list[InterruptEvent] = Field(default_factory=list)
    
    # Flag to indicate if execution is suspended
    is_suspended: bool = False
    
    # ID of the current interrupt (if suspended)
    current_interrupt_id: Optional[str] = None
    
    def create_interrupt(
        self,
        interrupt_type: InterruptType,
        node_name: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> InterruptEvent:
        """
        Create a new interrupt and suspend execution.
        
        Args:
            interrupt_type: Type of interrupt
            node_name: Name of the node creating the interrupt
            message: Explanation for the interrupt
            data: Additional context data
            
        Returns:
            The created InterruptEvent
        """
        interrupt = InterruptEvent(
            type=interrupt_type,
            node_name=node_name,
            message=message,
            data=data or {}
        )
        
        self.interrupts.append(interrupt)
        self.is_suspended = True
        self.current_interrupt_id = interrupt.id
        
        return interrupt
    
    def respond_to_interrupt(
        self,
        interrupt_id: str,
        status: InterruptStatus,
        response: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Respond to an interrupt and resume execution.
        
        Args:
            interrupt_id: ID of the interrupt to respond to
            status: Response status (approved, rejected, etc.)
            response: Human's response message
            data: Additional data from human
            
        Returns:
            True if interrupt was found and updated, False otherwise
        """
        for interrupt in self.interrupts:
            if interrupt.id == interrupt_id:
                interrupt.respond(status, response, data)
                
                # Resume execution if this is the current interrupt
                if self.current_interrupt_id == interrupt_id:
                    self.is_suspended = False
                    self.current_interrupt_id = None
                
                return True
        
        return False
    
    def get_pending_interrupts(self) -> list[InterruptEvent]:
        """Get all pending interrupts."""
        return [i for i in self.interrupts if i.status == InterruptStatus.PENDING]
    
    def get_current_interrupt(self) -> Optional[InterruptEvent]:
        """Get the currently active interrupt."""
        if self.current_interrupt_id:
            for interrupt in self.interrupts:
                if interrupt.id == self.current_interrupt_id:
                    return interrupt
        return None


# Import uuid for ID generation
import uuid


class HITLDecisionState(HITLState):
    """
    Extended DecisionState with HITL capabilities.
    
    This combines the standard DecisionState fields with HITL functionality.
    """
    
    # All DecisionState fields would be included here
    # For now, we're keeping this as a mixin pattern
    pass
