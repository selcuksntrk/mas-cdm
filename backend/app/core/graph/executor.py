"""

This module defines the decision-making graph and provides functionality
to execute the multi-agent decision workflow. The graph orchestrates
the flow between agent nodes and evaluator nodes to make informed decisions.

Each agent node's output is validated by its corresponding evaluator node,
which either advances the workflow or loops back with feedback for improvement.

"""

from pydantic_graph import Graph

from backend.app.models.domain import DecisionState
from backend.app.core.communication.message_broker import message_broker
from backend.app.core.graph.nodes import (
    GetDecision,
    IdentifyTrigger,
    AnalyzeRootCause,
    ScopeDefinition,
    Drafting,
    EstablishGoals,
    IdentifyInformationNeeded,
    RetrieveInformationNeeded,
    UpdateDraft,
    GenerationOfAlternatives,
    Result,
    # Evaluator nodes
    Evaluate_IdentifyTrigger,
    Evaluate_AnalyzeRootCause,
    Evaluate_ScopeDefinition,
    Evaluate_Drafting,
    Evaluate_EstablishGoals,
    Evaluate_IdentifyInformationNeeded,
    Evaluate_RetrieveInformationNeeded,
    Evaluate_UpdateDraft,
    Evaluate_GenerationOfAlternatives,
    Evaluate_Result,
)


# Graph Nodes Mapping
GRAPH_NODES = [
    # Entry point
    GetDecision,
    
    # Decision analysis phase
    IdentifyTrigger,
    Evaluate_IdentifyTrigger,
    AnalyzeRootCause,
    Evaluate_AnalyzeRootCause,
    ScopeDefinition,
    Evaluate_ScopeDefinition,
    
    # Decision drafting phase
    Drafting,
    Evaluate_Drafting,
    EstablishGoals,
    Evaluate_EstablishGoals,
    
    # Information gathering phase
    IdentifyInformationNeeded,
    Evaluate_IdentifyInformationNeeded,
    RetrieveInformationNeeded,
    Evaluate_RetrieveInformationNeeded,
    UpdateDraft,
    Evaluate_UpdateDraft,
    
    # Alternatives and final decision phase
    GenerationOfAlternatives,
    Evaluate_GenerationOfAlternatives,
    Result,
    Evaluate_Result,
]

# Define the decision-making graph with all agent and evaluator nodes
decision_graph = Graph(
    nodes=GRAPH_NODES,
    state_type=DecisionState,
)


# Graph Executor
async def run_decision_graph(decision_query: str) -> DecisionState:
    """
    Execute the complete decision-making graph workflow.
    
    Args:
        decision_query: The user's decision request
        
    Returns:
        DecisionState: The final state containing all decision outputs
    """
    # Initialize state with user's decision query.
    # Only 'decision_requested' is set here because it is the input to the workflow;
    # other fields in DecisionState will be populated as the graph executes each node.
    initial_state = DecisionState(decision_requested=decision_query)
    
    # Run the graph starting from GetDecision node
    final_state = await decision_graph.run(GetDecision(), state=initial_state)
    
    return final_state


# Get Mermaid representation of the graph
def get_graph_mermaid() -> str:
    """
    Generate a Mermaid diagram representation of the decision graph.
    
    Returns:
        str: Mermaid diagram code
        
    Example:
        >>> mermaid_code = get_graph_mermaid()
        >>> print(mermaid_code)
    """
    return decision_graph.mermaid_code(start_node=GetDecision())


# Get graph structure as a dictionary
def get_graph_structure() -> dict:
    """
    Get the structure of the decision graph including all nodes.
    
    Returns:
        dict: Graph structure information
    """
    nodes = []
    for node in GRAPH_NODES:
        # All nodes in GRAPH_NODES are classes
        node_name = node.__name__
        nodes.append({
            "name": node_name,
            "type": "evaluator" if node_name.startswith("Evaluate") else "agent",
        })
    
    return {
        "total_nodes": len(GRAPH_NODES),
        "agent_nodes": len([n for n in nodes if n["type"] == "agent"]),
        "evaluator_nodes": len([n for n in nodes if n["type"] == "evaluator"]),
        "nodes": nodes,
        "state_type": "DecisionState",
    }