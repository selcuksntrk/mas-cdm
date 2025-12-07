"""
Health Check Routes

Simple health check endpoints to verify API status.
"""

from fastapi import APIRouter

from backend.app.models.responses import HealthResponse


router = APIRouter(
    prefix="",
    tags=["health"],
)


@router.get("/", response_model=HealthResponse)
async def root():
    """
    Root endpoint with API information.
    
    Returns basic information about the API and available endpoints.
    """
    return HealthResponse(
        status="healthy",
        message="Multi-Agent Decision Making API",
        version="0.2.0",
        endpoints={
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /graph/mermaid": "Get decision graph visualization",
            "POST /decisions/run": "Run decision process synchronously",
            "POST /decisions/start": "Start decision process asynchronously",
            "GET /decisions/status/{process_id}": "Get process status",
            "POST /decisions/cli": "Run with persistence (debug mode)",
            "DELETE /decisions/cleanup": "Clean up completed processes",
        }
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the API.
    """
    return HealthResponse(
        status="healthy",
        message="API is running",
        version="0.2.0"
    )
