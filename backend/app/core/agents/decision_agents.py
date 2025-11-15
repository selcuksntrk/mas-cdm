"""

Decision Agents Module - This module contains classes and functions related 
to decision-making agents within the MAS-CDM (Multi Agent System for Critical 
Decision Making) backend application.

"""


from pydantic_ai import Agent
from app.config import get_settings
from app.models.domain import ResultOutput
from app.utils.helpers import load_prompt


# Get applcation settings
settings = get_settings()



# Identify Trigger Agent
# This agent is responsible for identifying triggers based on input data.
identify_trigger_agent = Agent(
    model=settings.decision_model_name,
    system_prompt=load_prompt("identify_trigger_agent.md")
)


# Root Cause Analyzer Agent
# This agent analyzes the root causes of identified triggers.
root_cause_analyzer_agent = Agent(
    model=settings.decision_model_name,
    system_prompt=load_prompt("root_cause_analyzer_agent.txt")
)


# Scope Definition Agent
# This agent defines the boundaries and scope of the decision-making process.
scope_definition_agent = Agent(
    model=settings.decision_model_name,
    system_prompt=load_prompt("scope_definition_agent.txt")
)


# Drafting Agent
# Draft the initial decision based on analysis and defined scope.
drafting_agent = Agent(
    model=settings.decision_model_name,
    system_prompt=load_prompt("drafting_agent.txt")
)


# Establish Goals Agent
# Defines clear goals for the decision-making process.
establish_goals_agent = Agent(
    model=settings.decision_model_name,
    system_prompt=load_prompt("establish_goals_agent.txt")
)


# Identify Information Needed Agent
# Identifies the information required to make informed decisions.
identify_information_needed_agent = Agent(
    model=settings.decision_model_name,
    system_prompt=load_prompt("identify_information_needed_agent.txt")
)


# Retrieve information Needed Agent
# Retrieves the necessary information for decision-making.
retrieve_information_needed_agent = Agent(
    model=settings.decision_model_name,
    system_prompt=load_prompt("retrieve_information_needed_agent.txt")
)


# Draft Update Agent
# Updates the draft decision based on new information.
draft_update_agent = Agent(
    model=settings.decision_model_name,
    system_prompt=load_prompt("draft_update_agent.txt")
)


# Generation of Alternatives Agent
# Generates alternative options for decision-making.
generation_of_alternatives_agent = Agent(
    model=settings.decision_model_name,
    system_prompt=load_prompt("generation_of_alternatives_agent.txt")
)


# Result Agent
# Evaluates and presents the final decision outcome.
result_agent = Agent(
    model=settings.decision_model_name,
    system_prompt=load_prompt("result_agent.txt")
)


# Agents Registry for easy access and management
decision_agents_registry = {
    "identify_trigger_agent": identify_trigger_agent,
    "root_cause_analyzer_agent": root_cause_analyzer_agent,
    "scope_definition_agent": scope_definition_agent,
    "drafting_agent": drafting_agent,
    "establish_goals_agent": establish_goals_agent,
    "identify_information_needed_agent": identify_information_needed_agent,
    "retrieve_information_needed_agent": retrieve_information_needed_agent,
    "draft_update_agent": draft_update_agent,
    "generation_of_alternatives_agent": generation_of_alternatives_agent,
    "result_agent": result_agent,
}


# Get agent method
def get_decision_agent(agent_name: str) -> Agent:
    """Retrieve a decision agent by name from the registry.

    Args:
        agent_name (str): The name of the agent to retrieve.
    
    Returns:
        Agent: The requested decision agent.
    """
    
    if agent_name not in decision_agents_registry:
        raise ValueError(
            f"Agent '{agent_name}' not found in the decision agents registry. "
            f"Available agents: {list(decision_agents_registry.keys())}"
        )
    
    return decision_agents_registry[agent_name]


# List agents method
def list_decision_agents() -> list[str]:
    """List all available decision agent names.

    Returns:
        list[str]: A list of decision agent names.
    """
    return list(decision_agents_registry.keys())
