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

from app.config import get_settings
from app.models.domain import EvaluationOutput
from app.utils.helpers import load_prompt


# Get application settings
settings = get_settings()



# Identify Trigger Evaluator Agent
# This agent evaluates the result of the Identify Trigger Agent.
identify_trigger_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("identify_trigger_agent_evaluator.txt"),
)


# Root Cause Analyzer Evaluator Agent
# This agent evaluates the result of the Root Cause Analyzer Agent.
root_cause_analyzer_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("root_cause_analyzer_agent_evaluator.txt"),
)


# SCope Definition Evaluator Agent
# This agent evaluates the result of the Scope Definition Agent.
scope_definition_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("scope_definition_agent_evaluator.txt"),
)


# Drafting Evaluator Agent
# This agent evaluates the result of the Drafting Agent.
drafting_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("drafting_agent_evaluator.txt"),
)


# Establish Goals Evaluator Agent
# This agent evaluates the result of the Establish Goals Agent.
establish_goals_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("establish_goals_agent_evaluator.txt"),
)


# Identify Information Needed Evaluator Agent
# This agent evaluates the result of the Identify Information Needed Agent.
identify_information_needed_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("identify_information_needed_agent_evaluator.txt"),
)


# Retrieve Information Needed Evaluator Agent
# This agent evaluates the result of the Retrieve Information Needed Agent.
retrieve_information_needed_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("retrieve_information_needed_agent_evaluator.txt"),
)


# Draft Update Evaluator Agent
# This agent evaluates the result of the Draft Update Agent.
draft_update_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("draft_update_agent_evaluator.txt"),
)


# Generation of Alternatives Evaluator Agent
# This agent evaluates the result of the Generation of Alternatives Agent.
generation_of_alternatives_agent_evaluator = Agent(
    model=settings.evaluator_model_name,
    output_type=EvaluationOutput,
    system_prompt=load_prompt("generation_of_alternatives_agent_evaluator.txt"),
)


# Evaluator Agents Registry
# A registry to map evaluator agent names to their instances.
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
}


# Retrieve evaluator agent by name
def get_evaluator_agent(agent_name: str) -> Agent:
    """
    Retrieve an evaluator agent by its name from the registry.

    Args:
        agent_name (str): The name of the evaluator agent to retrieve.
    
    Returns:
        Agent: The evaluator agent instance.
    """
    
    if agent_name not in evaluator_agents_registry:
        raise ValueError(
            f"Agent '{agent_name}' not found in the evaluator agents registry. "
            f"Available agents: {list(evaluator_agents_registry.keys())}"
        )
    
    return evaluator_agents_registry[agent_name]


# List evaluator agents method
def list_evaluator_agents() -> list[str]:
    """
    List all available evaluator agent names.

    Returns:
        list[str]: A list of evaluator agent names.
    """
    return list(evaluator_agents_registry.keys())