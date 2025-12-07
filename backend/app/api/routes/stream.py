"""
Streaming API Routes

This module provides Server-Sent Events (SSE) endpoints for real-time feedback
during long-running agent execution.
"""

import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.app.models.requests import DecisionRequest
from backend.app.models.domain import DecisionState
from backend.app.services import DecisionService
from backend.app.core.graph import decision_graph
from backend.app.core.graph.nodes import GetDecision


router = APIRouter(
    prefix="/stream",
    tags=["streaming"]
)

decision_service = DecisionService()


class StreamEvent(BaseModel):
    """Model for streaming events."""
    event_type: str
    data: dict
    timestamp: str = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.timestamp is None:
            from datetime import datetime, UTC
            self.timestamp = datetime.now(UTC).isoformat()


async def decision_stream_generator(decision_query: str) -> AsyncGenerator[str, None]:
    """
    Generator function for streaming decision-making progress.
    
    Yields Server-Sent Events (SSE) formatted messages for each step of the
    decision-making process.
    
    Args:
        decision_query: The user's decision request
        
    Yields:
        SSE formatted strings with progress updates
    """
    try:
        # Send start event
        event = StreamEvent(
            event_type="start",
            data={"message": "Starting decision-making process", "query": decision_query}
        )
        yield f"data: {event.model_dump_json()}\n\n"
        
        # Create initial state
        state = DecisionState(decision_requested=decision_query)
        first_node = GetDecision()
        
        # Send phase updates as we progress through the graph
        step_count = 0
        
        # Run the graph with iteration to capture intermediate steps
        async for current_state in run_graph_with_streaming(first_node, state):
            step_count += 1
            
            # Determine the current phase based on state
            phase = determine_current_phase(current_state)
            
            # Create progress event
            event = StreamEvent(
                event_type="progress",
                data={
                    "step": step_count,
                    "phase": phase,
                    "trigger": current_state.trigger if current_state.trigger else None,
                    "root_cause": current_state.root_cause if current_state.root_cause else None,
                    "scope_definition": current_state.scope_definition if current_state.scope_definition else None,
                    "decision_drafted": current_state.decision_drafted if current_state.decision_drafted else None,
                    "goals": current_state.goals if current_state.goals else None,
                    "alternatives": current_state.alternatives if current_state.alternatives else None,
                    "selected_decision": current_state.result if current_state.result else None,
                }
            )
            yield f"data: {event.model_dump_json()}\n\n"
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.1)
        
        # Extract final results
        result = decision_service.extract_full_result(state)
        
        # Send completion event
        event = StreamEvent(
            event_type="complete",
            data={
                "message": "Decision-making process complete",
                "result": result
            }
        )
        yield f"data: {event.model_dump_json()}\n\n"
        
    except Exception as e:
        # Send error event
        event = StreamEvent(
            event_type="error",
            data={
                "message": str(e),
                "error_type": type(e).__name__
            }
        )
        yield f"data: {event.model_dump_json()}\n\n"


async def run_graph_with_streaming(first_node, state: DecisionState):
    """
    Run the decision graph and yield state at each step.
    
    This is a helper function that executes the graph and yields intermediate
    states for streaming progress updates.
    
    Args:
        first_node: The starting node of the graph
        state: Initial decision state
        
    Yields:
        DecisionState at each step of execution
    """
    # For now, we'll simulate streaming by running the graph and yielding
    # at key milestones. In a more advanced implementation, you'd modify
    # the graph executor to support streaming callbacks.
    
    # Run a partial execution and yield state
    yield state
    
    # Execute the graph completely
    await decision_graph.run(first_node, state=state)
    
    # Yield final state
    yield state


def determine_current_phase(state: DecisionState) -> str:
    """
    Determine the current phase of the decision-making process based on state.
    
    Args:
        state: Current decision state
        
    Returns:
        String representing the current phase
    """
    if state.result:
        return "Decision"
    elif state.alternatives:
        return "Alternatives"
    elif state.complementary_info:
        return "Information Retrieval"
    elif state.goals:
        return "Goal Establishment"
    elif state.decision_drafted:
        return "Drafting"
    elif state.scope_definition:
        return "Scope Definition"
    elif state.root_cause:
        return "Root Cause Analysis"
    elif state.trigger:
        return "Trigger Identification"
    else:
        return "Initialization"


@router.post("/decisions/stream")
async def stream_decision(request: DecisionRequest):
    """
    Stream decision-making progress in real-time using Server-Sent Events (SSE).
    
    This endpoint provides real-time updates as the decision-making process progresses.
    Use this for long-running decisions where you want to show live progress to users.
    
    The stream will emit events with the following types:
    - `start`: Process initialization
    - `progress`: Intermediate updates with current phase and available results
    - `complete`: Final results
    - `error`: If an error occurs
    
    Args:
        request: DecisionRequest containing the decision query
        
    Returns:
        StreamingResponse with Server-Sent Events
        
    Example:
        ```
        POST /stream/decisions/stream
        {
            "decision_query": "Should I switch careers?"
        }
        
        # Response (SSE stream):
        data: {"event_type":"start","data":{"message":"Starting..."},"timestamp":"..."}
        
        data: {"event_type":"progress","data":{"step":1,"phase":"Trigger Identification",...},"timestamp":"..."}
        
        data: {"event_type":"complete","data":{"message":"Complete","result":{...}},"timestamp":"..."}
        ```
        
    Usage with JavaScript:
        ```javascript
        const eventSource = new EventSource('/stream/decisions/stream');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data.event_type, data.data);
        };
        ```
    """
    try:
        # Validate the decision query
        decision_service.validate_decision_query(request.decision_query)
        
        # Return streaming response
        return StreamingResponse(
            decision_stream_generator(request.decision_query),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering in nginx
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting streaming process: {str(e)}"
        )


@router.get("/health")
async def stream_health():
    """
    Health check endpoint for streaming service.
    
    Returns:
        Status message
    """
    return {"status": "healthy", "service": "streaming"}
