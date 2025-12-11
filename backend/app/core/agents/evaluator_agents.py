"""

Evaluator Agents Module - This module contains classes and functions related 
to evaluator agents within the MAS-CDM (Multi Agent System for Critical 
Decision Making) backend application.

Evaluator agents are responsible for assessing and validating decisions made 
by decision agents to ensure they meet predefined criteria and standards.

Evaluator agents use EvaluationOutput to provide:
- Boolean pass/fail status
- Detailed feedback
- Suggestions for improvement

"""


from pydantic_ai import Agent

from backend.app.config import get_settings
from backend.app.models.domain import EvaluationOutput, AgentMetadata
from backend.app.utils.helpers import load_prompt
from backend.app.core.agents.registry import agent_registry


# Get application settings
settings = get_settings()


def _evaluator_meta(name: str, description: str) -> AgentMetadata:
    """Create metadata for evaluator agents."""
    return AgentMetadata(
        name=name,
        role="evaluator",
        description=description,
        model=settings.evaluator_model_name,
        tools=[],  # Evaluators don't use tools
    )



# Identify Trigger Evaluator Agent
# This agent evaluates the result of the Identify Trigger Agent.
identify_trigger_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("identify_trigger_agent_evaluator.txt"),
)
agent_registry.register(
    "identify_trigger_agent_evaluator",
    identify_trigger_agent_evaluator,
    _evaluator_meta("identify_trigger_agent_evaluator", "Evaluates trigger identification"),
)


# Root Cause Analyzer Evaluator Agent
# This agent evaluates the result of the Root Cause Analyzer Agent.
root_cause_analyzer_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("root_cause_analyzer_agent_evaluator.txt"),
)
agent_registry.register(
    "root_cause_analyzer_agent_evaluator",
    root_cause_analyzer_agent_evaluator,
    _evaluator_meta("root_cause_analyzer_agent_evaluator", "Evaluates root cause analysis"),
)


# SCope Definition Evaluator Agent
# This agent evaluates the result of the Scope Definition Agent.
scope_definition_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("scope_definition_agent_evaluator.txt"),
)
agent_registry.register(
    "scope_definition_agent_evaluator",
    scope_definition_agent_evaluator,
    _evaluator_meta("scope_definition_agent_evaluator", "Evaluates scope definition"),
)


# Drafting Evaluator Agent
# This agent evaluates the result of the Drafting Agent.
drafting_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("drafting_agent_evaluator.txt"),
)
agent_registry.register(
    "drafting_agent_evaluator",
    drafting_agent_evaluator,
    _evaluator_meta("drafting_agent_evaluator", "Evaluates decision drafts"),
)


# Establish Goals Evaluator Agent
# This agent evaluates the result of the Establish Goals Agent.
establish_goals_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("establish_goals_agent_evaluator.txt"),
)
agent_registry.register(
    "establish_goals_agent_evaluator",
    establish_goals_agent_evaluator,
    _evaluator_meta("establish_goals_agent_evaluator", "Evaluates goal establishment"),
)


# Identify Information Needed Evaluator Agent
# This agent evaluates the result of the Identify Information Needed Agent.
identify_information_needed_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("identify_information_needed_agent_evaluator.txt"),
)
agent_registry.register(
    "identify_information_needed_agent_evaluator",
    identify_information_needed_agent_evaluator,
    _evaluator_meta("identify_information_needed_agent_evaluator", "Evaluates information needs"),
)


# Retrieve Information Needed Evaluator Agent
# This agent evaluates the result of the Retrieve Information Needed Agent.
retrieve_information_needed_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("retrieve_information_needed_agent_evaluator.txt"),
)
agent_registry.register(
    "retrieve_information_needed_agent_evaluator",
    retrieve_information_needed_agent_evaluator,
    _evaluator_meta("retrieve_information_needed_agent_evaluator", "Evaluates retrieved information"),
)


# Draft Update Evaluator Agent
# This agent evaluates the result of the Draft Update Agent.
draft_update_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("draft_update_agent_evaluator.txt"),
)
agent_registry.register(
    "draft_update_agent_evaluator",
    draft_update_agent_evaluator,
    _evaluator_meta("draft_update_agent_evaluator", "Evaluates draft updates"),
)


# Generation of Alternatives Evaluator Agent
# This agent evaluates the result of the Generation of Alternatives Agent.
generation_of_alternatives_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("generation_of_alternatives_agent_evaluator.txt"),
)
agent_registry.register(
    "generation_of_alternatives_agent_evaluator",
    generation_of_alternatives_agent_evaluator,
    _evaluator_meta("generation_of_alternatives_agent_evaluator", "Evaluates generated alternatives"),
)

# Result Evaluator Agent
# This agent evaluates the result of the Result Agent.
result_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("result_agent_evaluator.txt"),
)
agent_registry.register(
    "result_agent_evaluator",
    result_agent_evaluator,
    _evaluator_meta("result_agent_evaluator", "Evaluates final results"),
)


# Legacy evaluator_agents_registry - DEPRECATED, use agent_registry instead
# Kept for backwards compatibility
evaluator_agents_registry = {
    "identify_trigger_agent_evaluator": identify_trigger_agent_evaluator,
    "root_cause_analyzer_agent_evaluator": root_cause_analyzer_agent_evaluator,
    "scope_definition_agent_evaluator": scope_definition_agent_evaluator,
    "drafting_agent_evaluator": drafting_agent_evaluator,
    "establish_goals_agent_evaluator": establish_goals_agent_evaluator,
    "identify_information_needed_agent_evaluator": identify_information_needed_agent_evaluator,
    "retrieve_information_needed_agent_evaluator": retrieve_information_needed_agent_evaluator,
    "draft_update_agent_evaluator": draft_update_agent_evaluator,
    "generation_of_alternatives_agent_evaluator": generation_of_alternatives_agent_evaluator,
    "result_agent_evaluator": result_agent_evaluator,
}


# Retrieve evaluator agent by name
def get_evaluator_agent(agent_name: str) -> Agent:
    """
    Retrieve an evaluator agent by its name from the unified registry.

    Args:
        agent_name (str): The name of the evaluator agent to retrieve.
    
    Returns:
        Agent: The evaluator agent instance.
    
    Note:
        This function now uses the unified agent_registry instead of the 
        separate evaluator_agents_registry. Both registries are kept in sync.
    """
    agent = agent_registry.get(agent_name)
    if agent is None:
        # Fallback to legacy registry for backwards compatibility
        if agent_name in evaluator_agents_registry:
            return evaluator_agents_registry[agent_name]
        available = [name for name in agent_registry.list() if "evaluator" in name]
        raise ValueError(
            f"Agent '{agent_name}' not found. "
            f"Available evaluator agents: {available}"
        )
    return agent


# List evaluator agents method
def list_evaluator_agents() -> list[str]:
    """
    List all available evaluator agent names.

    Returns:
        list[str]: A list of evaluator agent names.
    """
    return list(evaluator_agents_registry.keys())