"""

Decisions API Routes

This module defines the API routes related to decisions.

"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks

from backend.app.models.requests import DecisionRequest
from backend.app.models.responses import (
    DecisionResponse,
    ProcessStartResponse,
    ProcessStatusResponse,
    ErrorResponse
)
from backend.app.services import DecisionService, get_process_manager

router = APIRouter(
    prefix="/decisions",
    tags=["decisions"]
)

# Initialize Decision Service
decision_service = DecisionService()

@router.post("/run", response_model=DecisionResponse)
async def run_decision_sync(request: DecisionRequest):
    """
    Run a complete decision-making process synchronously.
    
    This endpoint will block until the entire decision-making process is complete.
    Use this when you need immediate results.
    
    Args:
        request: DecisionRequest containing the decision query
        
    Returns:
        DecisionResponse: Complete decision results with all phases
        
    Raises:
        HTTPException: 400 for validation errors, 500 for processing errors
        
    Example:
        ```
        POST /decisions/run
        {
            "decision_query": "Should I switch careers from engineering to product management?"
        }
        
        Response:
        {
            "selected_decision": "...",
            "selected_decision_comment": "...",
            "alternative_decision": "...",
            "alternative_decision_comment": "...",
            "trigger": "...",
            "root_cause": "...",
            ...
        }
        ```
    """
    
    try:
        # Validate the decision query
        decision_service.validate_decision_query(request.decision_query)
        
        # Run the decision process
        state = await decision_service.run_decision(request.decision_query)
        
        # Extract and return the results
        result = decision_service.extract_full_result(state)
        
        return DecisionResponse(
            selected_decision=result["selected_decision"],
            selected_decision_comment=result["selected_decision_comment"],
            alternative_decision=result["alternative_decision"],
            alternative_decision_comment=result["alternative_decision_comment"],
            trigger=result["trigger"],
            root_cause=result["root_cause"],
            scope_definition=result["scope_definition"],
            decision_drafted=result["decision_drafted"],
            goals=result["goals"],
            complementary_info=result.get("complementary_info", ""),
            decision_draft_updated=result.get("decision_draft_updated", ""),
            alternatives=result.get("alternatives", ""),
        )
    
    except ValueError as e:
        # Validation error
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Processing error
        raise HTTPException(
            status_code=500,
            detail=f"Error running decision process: {str(e)}"
        )


@router.post("/start", response_model=ProcessStartResponse)
async def start_decision_async(request: DecisionRequest, background_tasks: BackgroundTasks):
    """
    Start a decision-making process asynchronously.
    
    This endpoint immediately returns a process_id that can be used to check status.
    The actual decision-making runs in the background.
    
    ASYNC PATTERN EXPLAINED:
    ========================
    1. Create process (stored in repository)
    2. Return process_id immediately
    3. Run actual decision in background
    4. Client polls /status/{process_id} for updates
    
    This allows:
    - Responsive API (no long waits)
    - Multiple concurrent processes
    - Scalable architecture
    
    Args:
        request: DecisionRequest containing the decision query
        background_tasks: FastAPI background tasks (injected)
        
    Returns:
        ProcessStartResponse: Process ID and status
        
    Raises:
        HTTPException: 400 for validation errors, 500 for processing errors
        
    Example:
        ```
        POST /decisions/start
        {
            "decision_query": "Should I switch careers?"
        }
        
        Response:
        {
            "process_id": "process_abc123def456",
            "status": "started",
            "message": "Decision-making process started in background"
        }
        ```
    """
    try:
        # Validate the decision query
        decision_service.validate_decision_query(request.decision_query)
        
        # Get process manager
        manager = get_process_manager()
        
        # Create new process (now async with repository)
        process_info = await manager.create_process(request.decision_query)
        
        # Add background task to execute the process
        background_tasks.add_task(
            manager.execute_process,
            process_info.process_id,
            request.decision_query
        )
        
        return ProcessStartResponse(
            process_id=process_info.process_id,
            status=process_info.status,
            message="Decision-making process started in background"
        )
    
    except ValueError as e:
        # Validation error
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Processing error
        raise HTTPException(
            status_code=500,
            detail=f"Error starting decision process: {str(e)}"
        )
        
@router.get("/status/{process_id}", response_model=ProcessStatusResponse)
async def get_decision_status(process_id: str):
    """
    Get the status of a running decision-making process.
    
    POLLING PATTERN:
    ================
    Clients should poll this endpoint periodically:
    - Every 1-2 seconds while status="running"
    - Stop polling when status="completed" or "failed"
    
    Alternative approaches:
    - WebSockets (real-time updates)
    - Server-Sent Events (SSE)
    - Webhooks (callback when done)
    
    Current approach (polling) is simple and works well for moderate scale.
    
    Args:
        process_id: The process identifier returned from /decisions/start
        
    Returns:
        ProcessStatusResponse: Status and result (if completed)
        
    Raises:
        HTTPException: 404 if process not found
        
    Example:
        ```
        GET /decisions/status/process_abc123def456
        
        Response (running):
        {
            "process_id": "process_abc123def456",
            "status": "running"
        }
        
        Response (completed):
        {
            "process_id": "process_abc123def456",
            "status": "completed",
            "result": {
                "selected_decision": "...",
                ...
            }
        }
        ```
    """
    # Get process manager
    manager = get_process_manager()
    
    # Check if process exists (now async)
    if not await manager.process_exists(process_id):
        raise HTTPException(status_code=404, detail="Process not found")
    
    # Get process info (now async)
    process_info = await manager.get_process(process_id)
    
    # Build response
    response = ProcessStatusResponse(
        process_id=process_id,
        status=process_info.status,
        created_at=process_info.created_at,
        completed_at=process_info.completed_at,
        current_step=process_info.current_step,
        completed_steps=process_info.completed_steps
    )
    
    if process_info.status == "completed" and process_info.result:
        # Extract result summary
        result = decision_service.extract_full_result(process_info.result)
        response.result = result
    
    elif process_info.status == "failed":
        response.error = process_info.error
    
    return response


@router.post("/cli")
async def run_decision_cli(request: DecisionRequest):
    """
    Run a decision-making process with file-based persistence (CLI mode).
    
    This endpoint uses file-based persistence to save the graph state.
    Useful for debugging or analyzing the execution flow.
    
    Args:
        request: DecisionRequest containing the decision query
        
    Returns:
        dict: Execution results including history and persistence file path
        
    Raises:
        HTTPException: 400 for validation errors, 500 for processing errors
        
    Example:
        ```
        POST /decisions/cli
        {
            "decision_query": "Should I switch careers?"
        }
        
        Response:
        {
            "status": "completed",
            "final_state": {...},
            "execution_history": ["GetDecision", "IdentifyTrigger", ...],
            "persistence_file": "/path/to/decision_graph.json"
        }
        ```
    """
    try:
        # Validate the decision query
        decision_service.validate_decision_query(request.decision_query)
        
        # Run with persistence
        persistence_file = Path('decision_graph.json')
        state, execution_history = await decision_service.run_decision_with_persistence(
            request.decision_query,
            persistence_file
        )
        
        # Extract result summary
        result_summary = decision_service.extract_result_summary(state)
        
        return {
            "status": "completed",
            "final_state": result_summary,
            "execution_history": execution_history,
            "persistence_file": str(persistence_file.absolute())
        }
    
    except ValueError as e:
        # Validation error
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Processing error
        raise HTTPException(
            status_code=500,
            detail=f"Error running CLI mode: {str(e)}"
        )
        
@router.delete("/cleanup")
async def cleanup_processes():
    """
    Clean up completed and failed processes from storage.
    
    CLEANUP STRATEGY:
    =================
    This endpoint removes old completed/failed processes to:
    - Free up memory (in-memory repository)
    - Free up Redis memory (Redis repository)
    - Improve query performance
    
    PRODUCTION RECOMMENDATION:
    ==========================
    Don't rely on manual cleanup. Instead:
    1. Add periodic cleanup job (APScheduler, Celery Beat)
    2. Run cleanup daily or hourly based on process volume
    3. Monitor cleanup metrics
    
    Example periodic cleanup:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        
        scheduler = AsyncIOScheduler()
        
        async def cleanup_job():
            manager = get_process_manager()
            await manager.cleanup_completed(older_than_hours=24)
        
        scheduler.add_job(cleanup_job, 'interval', hours=1)
        scheduler.start()
    
    Returns:
        dict: Cleanup statistics
        
    Example:
        ```
        DELETE /decisions/cleanup
        
        Response:
        {
            "status": "success",
            "message": "Cleaned up 5 completed/failed processes",
            "remaining_processes": 2,
            "stats": {
                "total": 2,
                "running": 2,
                "completed": 0,
                "failed": 0
            }
        }
        ```
    """
    try:
        # Get process manager
        manager = get_process_manager()
        
        # Clean up completed/failed processes (now async, with 24-hour threshold)
        cleaned_count = await manager.cleanup_completed(older_than_hours=24)
        
        # Get updated stats (now async)
        stats = await manager.get_stats()
        
        return {
            "status": "success",
            "message": f"Cleaned up {cleaned_count} completed/failed processes",
            "removed_count": cleaned_count,
            "remaining_processes": stats["total"],
            "stats": stats
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during cleanup: {str(e)}"
        )


@router.get("/processes")
async def list_processes():
    """
    List all tracked processes with their status.
    
    OBSERVABILITY:
    ==============
    This endpoint is crucial for:
    - Monitoring system health
    - Debugging issues
    - Understanding system load
    
    SCALING CONSIDERATIONS:
    =======================
    With many processes, this could be slow/expensive:
    - Redis: KEYS command scans all keys
    - In-memory: Iterates all processes
    
    Solutions:
    1. Add pagination (offset/limit)
    2. Add filtering (status, date range)
    3. Cache results with short TTL
    4. Use separate counters for stats
    
    Example paginated version:
        @router.get("/processes")
        async def list_processes(
            skip: int = 0,
            limit: int = 100,
            status: Optional[str] = None
        ):
            # Implementation with pagination
    
    Returns:
        dict: Process statistics and list of all processes
        
    Example:
        ```
        GET /decisions/processes
        
        Response:
        {
            "stats": {
                "total": 10,
                "running": 2,
                "completed": 7,
                "failed": 1
            },
            "processes": [...]
        }
        ```
    """
    try:
        # Get process manager
        manager = get_process_manager()
        
        # Get all processes (now async)
        processes = await manager.get_all_processes()
        
        # Get stats (now async)
        stats = await manager.get_stats()
        
        return {
            "stats": stats,
            "processes": [
                {
                    "process_id": p.process_id,
                    "status": p.status,
                    "created_at": p.created_at,
                    "completed_at": p.completed_at,
                    "has_error": p.error is not None,
                    # Only include result summary for completed processes
                    "has_result": p.result is not None
                }
                for p in processes
            ]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing processes: {str(e)}"
        )
