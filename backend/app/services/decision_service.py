"""
Decision Service

This service orchestrates the decision-making workflow using the graph executor.
It provides high-level operations for running decision processes both synchronously
and asynchronously.
"""


from pathlib import Path
from typing import Optional

from pydantic_graph import End
from pydantic_graph.persistence.file import FileStatePersistence

from backend.app.models.domain import DecisionState
from backend.app.core.graph import decision_graph, run_decision_graph
from backend.app.core.graph.nodes import GetDecision



# Decision Service
class DecisionService:
    """
    Service for orchestrating decision-making processes.
    
    This service provides methods to:
    - Run complete decision workflows synchronously
    - Execute decisions with persistence for debugging
    - Extract decision results from state
    """
    
    async def run_decision(self, decision_query: str) -> DecisionState:
        """
        Run a complete decision-making process synchronously.
        
        Args:
            decision_query: The user's decision request
            
        Returns:
            DecisionState: The final state with all decision outputs
            
        Raises:
            Exception: If the decision process fails
            
        Example:
            >>> service = DecisionService()
            >>> state = await service.run_decision("Should I switch careers?")
            >>> print(state.result)
        """
        # Create initial state with decision query
        state = DecisionState(decision_requested=decision_query)
        
        # Create first node (GetDecision will skip prompting since query is set)
        first_node = GetDecision()
        
        # Run the graph
        await decision_graph.run(first_node, state=state)
        
        return state
    
    async def run_decision_with_persistence(
        self,
        decision_query: str,
        persistence_file: Optional[Path] = None
    ) -> tuple[DecisionState, list[str]]:
        """
        Run a decision-making process with file-based persistence.
        
        Useful for debugging, resuming processes, or CLI mode.
        
        Args:
            decision_query: The user's decision request
            persistence_file: Path to persistence file (default: decision_graph.json)
            
        Returns:
            tuple[DecisionState, list[str]]: Final state and execution history
            
        Example:
            >>> service = DecisionService()
            >>> state, history = await service.run_decision_with_persistence(
            ...     "Should I switch careers?",
            ...     Path("debug/decision.json")
            ... )
            >>> print(f"Executed {len(history)} steps")
        """
        # Setup persistence
        if persistence_file is None:
            persistence_file = Path('decision_graph.json')
        
        persistence = FileStatePersistence(persistence_file)
        persistence.set_graph_types(decision_graph)
        
        # Create state and node
        node = GetDecision()
        state = DecisionState(decision_requested=decision_query)
        
        # Run the graph with persistence
        history = []
        async with decision_graph.iter(node, state=state, persistence=persistence) as run:
            while True:
                node = await run.next()
                history.append(type(node).__name__)
                
                if isinstance(node, End):
                    break
        
        # Load full history from persistence
        full_history = await persistence.load_all()
        execution_history = [str(e.node) for e in full_history]
        
        return state, execution_history
    
    @staticmethod
    def extract_result_summary(state: DecisionState) -> dict:
        """
        Extract a summary of the decision result from the final state.
        
        Args:
            state: The final DecisionState after workflow completion
            
        Returns:
            dict: Summary with selected decision and alternative
            
        Example:
            >>> service = DecisionService()
            >>> state = await service.run_decision("Should I switch careers?")
            >>> summary = service.extract_result_summary(state)
            >>> print(summary["selected_decision"])
        """
        return {
            "selected_decision": state.result,
            "selected_decision_comment": state.result_comment,
            "alternative_decision": state.best_alternative_result,
            "alternative_decision_comment": state.best_alternative_result_comment,
        }
    
    @staticmethod
    def extract_full_result(state: DecisionState) -> dict:
        """
        Extract the complete decision result including all intermediate steps.
        
        Args:
            state: The final DecisionState after workflow completion
            
        Returns:
            dict: Complete result with all decision phases
            
        Example:
            >>> service = DecisionService()
            >>> state = await service.run_decision("Should I switch careers?")
            >>> full_result = service.extract_full_result(state)
            >>> print(full_result["trigger"])
        """
        return {
            "selected_decision": state.result,
            "selected_decision_comment": state.result_comment,
            "alternative_decision": state.best_alternative_result,
            "alternative_decision_comment": state.best_alternative_result_comment,
            "trigger": state.trigger,
            "root_cause": state.root_cause,
            "scope_definition": state.scope_definition,
            "decision_drafted": state.decision_drafted,
            "goals": state.goals,
            "complementary_info": state.complementary_info,
            "decision_draft_updated": state.decision_draft_updated,
            "alternatives": state.alternatives,
        }
    
    @staticmethod
    def validate_decision_query(decision_query: str) -> bool:
        """
        Validate a decision query before processing.
        
        Args:
            decision_query: The decision query to validate
            
        Returns:
            bool: True if valid, raises exception if invalid
            
        Raises:
            ValueError: If the query is invalid
        """
        if not decision_query or not decision_query.strip():
            raise ValueError("Decision query cannot be empty")
        
        if len(decision_query.strip()) < 10:
            raise ValueError("Decision query must be at least 10 characters")
        
        if len(decision_query) > 1000:
            raise ValueError("Decision query must be less than 1000 characters")
        
        return True