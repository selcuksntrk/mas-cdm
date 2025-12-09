"""
Graph Visualization Routes

Endpoints for retrieving decision graph structure and visualization.
"""

from fastapi import APIRouter, HTTPException

from backend.app.models.responses import MermaidResponse, ErrorResponse
from backend.app.core.graph import get_graph_mermaid, get_graph_structure
from backend.app.core.graph.builder import graph_builder, GraphValidationError
from backend.app.models.requests import BuildGraphRequest
from backend.app.core.graph.nodes import GetDecision


router = APIRouter(
    prefix="/graph",
    tags=["graph"],
)


@router.get("/mermaid", response_model=MermaidResponse)
async def get_mermaid_diagram():
    """
    Get the Mermaid diagram code for the decision graph.
    
    This can be used to visualize the decision-making workflow.
    The Mermaid code can be rendered using any Mermaid-compatible tool.
    
    Returns:
        MermaidResponse: Contains the Mermaid diagram code
        
    Example:
        ```
        GET /graph/mermaid
        
        Response:
        {
            "mermaid_code": "graph TD\\n  A[GetDecision] --> B[IdentifyTrigger]\\n  ..."
        }
        ```
    """
    try:
        mermaid_code = get_graph_mermaid()
        return MermaidResponse(mermaid_code=mermaid_code)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating mermaid diagram: {str(e)}"
        )


@router.get("/structure")
async def get_structure():
    """
    Get the structure information of the decision graph.
    
    Returns metadata about the graph including node counts and types.
    
    Returns:
        dict: Graph structure information
        
    Example:
        ```
        GET /graph/structure
        
        Response:
        {
            "total_nodes": 19,
            "agent_nodes": 10,
            "evaluator_nodes": 9,
            "nodes": [...]
        }
        ```
    """
    try:
        return get_graph_structure()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting graph structure: {str(e)}"
        )


@router.post("/build")
async def build_graph(request: BuildGraphRequest):
    """
    Validate a custom graph specification and return the normalized structure.
    """
    try:
        built = graph_builder.build(request.nodes, request.edges)
        return built
    except GraphValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
