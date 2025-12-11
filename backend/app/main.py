"""
Main FastAPI Application

This is the entry point for the Multi-Agent Decision Making API.
It configures the FastAPI app with all routes, middleware, and settings.
"""

import logging
import time
from fastapi import FastAPI, Request, Security, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from backend.app.config import get_settings
from backend.app.api.routes import health, graph, decisions, stream, hitl, tools, agents
from backend.app.core.exceptions import (
    AgentExecutionError,
    GraphExecutionError,
    LLMRateLimitError,
    LLMAPIConnectionError,
    LLMContextWindowError,
    LLMContentPolicyError,
    ToolExecutionError,
    VectorStoreError,
    ConcurrencyError,
)
from backend.app.core.observability.tracer import get_tracer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Get application settings
settings = get_settings()


# API Key Security (optional, based on settings)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify API key for protected endpoints.
    
    Only enforced if settings.enable_auth is True.
    Raises HTTPException if auth is enabled and key is invalid.
    """
    # Skip auth if not enabled
    if not settings.enable_auth:
        return None
    
    # Auth is enabled - require valid key
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if not settings.api_auth_key:
        logger.error("Auth enabled but API_AUTH_KEY not configured!")
        raise HTTPException(
            status_code=500,
            detail="Authentication not properly configured",
        )
    
    if api_key != settings.api_auth_key:
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )
    
    return api_key


# Create FastAPI application
app = FastAPI(
    title="Multi-Agent Decision Making API",
    description="""
    A sophisticated multi-agent system for structured decision-making.
    
    This API orchestrates multiple AI agents through a graph-based workflow to help
    users make informed decisions. The system:
    
    - Identifies decision triggers and analyzes root causes
    - Defines scope and establishes SMART goals
    - Generates and evaluates alternatives
    - Provides reasoned recommendations with alternatives
    
    ## Features
    
    - **Synchronous Execution**: Get immediate results with `/decisions/run`
    - **Asynchronous Execution**: Start long-running processes with `/decisions/start`
    - **Process Tracking**: Monitor progress with `/decisions/status/{id}`
    - **Graph Visualization**: View workflow with `/graph/mermaid`
    - **Persistence Mode**: Debug with `/decisions/cli`
    
    ## Workflow Phases
    
    1. **Analysis**: Trigger identification, root cause analysis, scope definition
    2. **Drafting**: Initial decision draft and goal establishment
    3. **Information Gathering**: Identify and retrieve needed information
    4. **Alternatives**: Generate and evaluate alternatives
    5. **Decision**: Select best option with justification
    
    Each phase includes agent execution and evaluator validation with feedback loops.
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Configure CORS with environment-specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if settings.cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add tracing middleware
@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    """
    Middleware to add request tracing and performance monitoring.
    
    Captures:
    - Request ID for correlation
    - Request timing
    - Endpoint performance
    - Error tracking
    """
    # Generate request ID (always, for all requests)
    request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")
    
    # Skip detailed tracing for health checks and docs
    skip_detailed_tracing = request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]
    
    if skip_detailed_tracing or not settings.enable_tracing:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    
    tracer = get_tracer()
    start_time = time.time()
    
    try:
        # Log request start
        if tracer.enable_tracing:
            import logfire
            logfire.info(
                "Request started",
                method=request.method,
                path=request.url.path,
                request_id=request_id
            )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log request completion
        if tracer.enable_tracing:
            import logfire
            logfire.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                request_id=request_id,
                status_code=response.status_code,
                duration_seconds=duration
            )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Log request error
        if tracer.enable_tracing:
            import logfire
            logfire.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                request_id=request_id,
                error=str(e),
                duration_seconds=duration
            )
        
        raise


# Include routers
# Health check doesn't require auth
app.include_router(health.router)

# Protected routers - require API key if auth is enabled
app.include_router(
    graph.router,
    dependencies=[Depends(verify_api_key)] if settings.enable_auth else []
)
app.include_router(
    decisions.router,
    dependencies=[Depends(verify_api_key)] if settings.enable_auth else []
)
app.include_router(
    stream.router,
    dependencies=[Depends(verify_api_key)] if settings.enable_auth else []
)
app.include_router(
    hitl.router,
    dependencies=[Depends(verify_api_key)] if settings.enable_auth else []
)
app.include_router(
    tools.router,
    dependencies=[Depends(verify_api_key)] if settings.enable_auth else []
)
app.include_router(
    agents.router,
    dependencies=[Depends(verify_api_key)] if settings.enable_auth else []
)


# Specific exception handlers
@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """
    Handle data validation errors explicitly.
    
    Returns 400 Bad Request with validation error details.
    """
    logger.warning(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "type": "ValueError"
        }
    )


@app.exception_handler(AgentExecutionError)
async def agent_execution_exception_handler(request: Request, exc: AgentExecutionError):
    """Handle agent execution failures."""
    logger.error(f"Agent execution error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Agent Execution Error",
            "detail": str(exc),
            "type": "AgentExecutionError",
            "retry": True
        }
    )


@app.exception_handler(GraphExecutionError)
async def graph_execution_exception_handler(request: Request, exc: GraphExecutionError):
    """Handle graph execution failures."""
    logger.error(f"Graph execution error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Graph Execution Error",
            "detail": str(exc),
            "type": "GraphExecutionError",
            "retry": True
        }
    )


@app.exception_handler(LLMRateLimitError)
async def llm_rate_limit_exception_handler(request: Request, exc: LLMRateLimitError):
    """Handle LLM rate limit errors."""
    logger.warning(f"LLM rate limit exceeded: {str(exc)}")
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate Limit Exceeded",
            "detail": "The AI service rate limit has been exceeded. Please try again later.",
            "type": "LLMRateLimitError",
            "retry": True,
            "retry_after": 60  # Suggest retry after 60 seconds
        }
    )


@app.exception_handler(LLMAPIConnectionError)
async def llm_connection_exception_handler(request: Request, exc: LLMAPIConnectionError):
    """Handle LLM API connection errors."""
    logger.error(f"LLM API connection error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service Unavailable",
            "detail": "Cannot connect to AI service. Please try again later.",
            "type": "LLMAPIConnectionError",
            "retry": True
        }
    )


@app.exception_handler(LLMContextWindowError)
async def llm_context_window_exception_handler(request: Request, exc: LLMContextWindowError):
    """Handle LLM context window exceeded errors."""
    logger.warning(f"LLM context window exceeded: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Context Window Exceeded",
            "detail": "The input is too long for the AI model. Please reduce the input size.",
            "type": "LLMContextWindowError",
            "retry": False
        }
    )


@app.exception_handler(LLMContentPolicyError)
async def llm_content_policy_exception_handler(request: Request, exc: LLMContentPolicyError):
    """Handle LLM content policy violations."""
    logger.warning(f"LLM content policy violation: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Content Policy Violation",
            "detail": "The input violates the AI service content policy. Please modify your request.",
            "type": "LLMContentPolicyError",
            "retry": False
        }
    )


@app.exception_handler(ToolExecutionError)
async def tool_execution_exception_handler(request: Request, exc: ToolExecutionError):
    """Handle tool execution failures."""
    logger.error(f"Tool execution error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Tool Execution Error",
            "detail": str(exc),
            "type": "ToolExecutionError",
            "retry": True
        }
    )


@app.exception_handler(VectorStoreError)
async def vector_store_exception_handler(request: Request, exc: VectorStoreError):
    """Handle vector store errors."""
    logger.error(f"Vector store error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Vector Store Error",
            "detail": "An error occurred accessing the memory system.",
            "type": "VectorStoreError",
            "retry": True
        }
    )


@app.exception_handler(ConcurrencyError)
async def concurrency_exception_handler(request: Request, exc: ConcurrencyError):
    """Handle concurrency conflicts."""
    logger.warning(f"Concurrency error: {str(exc)}")
    return JSONResponse(
        status_code=409,
        content={
            "error": "Concurrency Conflict",
            "detail": "A concurrent modification was detected. Please retry your request.",
            "type": "ConcurrencyError",
            "retry": True
        }
    )


# Global exception handler (catch-all)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    
    Logs the full traceback but returns a safe error message to the client.
    This prevents leaking sensitive stack traces in production.
    """
    # Log the full exception with traceback for debugging
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # Determine if we're in debug mode
    is_debug = settings.debug
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            # Only include detailed error message in debug mode
            "detail": str(exc) if is_debug else "An unexpected error occurred. Please contact support.",
            "type": type(exc).__name__,
            "retry": False
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    
    Performs initialization tasks when the application starts.
    """
    print("=" * 60)
    print("Multi-Agent Decision Making API")
    print("=" * 60)
    print(f"Version: {app.version}")
    print(f"Docs: http://localhost:8001/docs")
    print(f"Model: {settings.model_name}")
    print(f"Evaluation Model: {settings.evaluation_model}")
    print("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.
    
    Performs cleanup tasks when the application shuts down.
    """
    print("\nShutting down Multi-Agent Decision Making APP...")


if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI app
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
