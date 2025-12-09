"""

Decision Agents Module - This module contains classes and functions related 
to decision-making agents within the MAS-CDM (Multi Agent System for Critical 
Decision Making) backend application.

"""


from pydantic_ai import Agent
from backend.app.config import get_settings
from backend.app.models.domain import ResultOutput, AgentMetadata
from backend.app.utils.helpers import load_prompt
from backend.app.core.tools import manager
from backend.app.core.agents.registry import agent_registry


# Get applcation settings
settings = get_settings()

# Tools exposed to decision agents for prompt conditioning or function calling
decision_agent_tools = manager.list_tools()


def _agent_model(agent_name: str) -> str:
    """Resolve per-agent model overrides from settings."""
    return settings.get_agent_model(agent_name)


def _meta(name: str, role: str, description: str) -> AgentMetadata:
    return AgentMetadata(
        name=name,
        role=role,
        description=description,
        model=_agent_model(name),
        tools=decision_agent_tools,
    )



# Identify Trigger Agent
# This agent is responsible for identifying triggers based on input data.
identify_trigger_agent = Agent(
    model=_agent_model("identify_trigger_agent"),
    system_prompt=load_prompt("identify_trigger_agent.txt")
)
agent_registry.register(
    "identify_trigger_agent",
    identify_trigger_agent,
    _meta(
        "identify_trigger_agent",
        "analyzer",
        "Identifies the trigger for the decision",
    ),
)


# Root Cause Analyzer Agent
# This agent analyzes the root causes of identified triggers.
root_cause_analyzer_agent = Agent(
    model=_agent_model("root_cause_analyzer_agent"),
    system_prompt=load_prompt("root_cause_analyzer_agent.txt")
)
agent_registry.register(
    "root_cause_analyzer_agent",
    root_cause_analyzer_agent,
    _meta(
        "root_cause_analyzer_agent",
        "analyzer",
        "Analyzes root causes of identified triggers",
    ),
)


# Scope Definition Agent
# This agent defines the boundaries and scope of the decision-making process.
scope_definition_agent = Agent(
    model=_agent_model("scope_definition_agent"),
    system_prompt=load_prompt("scope_definition_agent.txt")
)
agent_registry.register(
    "scope_definition_agent",
    scope_definition_agent,
    _meta(
        "scope_definition_agent",
        "planner",
        "Defines boundaries and scope of the decision",
    ),
)


# Drafting Agent
# Draft the initial decision based on analysis and defined scope.
drafting_agent = Agent(
    model=_agent_model("drafting_agent"),
    system_prompt=load_prompt("drafting_agent.txt")
)
agent_registry.register(
    "drafting_agent",
    drafting_agent,
    _meta(
        "drafting_agent",
        "writer",
        "Drafts the initial decision",
    ),
)


# Establish Goals Agent
# Defines clear goals for the decision-making process.
establish_goals_agent = Agent(
    model=_agent_model("establish_goals_agent"),
    system_prompt=load_prompt("establish_goals_agent.txt")
)
agent_registry.register(
    "establish_goals_agent",
    establish_goals_agent,
    _meta(
        "establish_goals_agent",
        "planner",
        "Establishes goals for the decision",
    ),
)


# Identify Information Needed Agent
# Identifies the information required to make informed decisions.
identify_information_needed_agent = Agent(
    model=_agent_model("identify_information_needed_agent"),
    system_prompt=load_prompt("identify_information_needed_agent.txt")
)
agent_registry.register(
    "identify_information_needed_agent",
    identify_information_needed_agent,
    _meta(
        "identify_information_needed_agent",
        "researcher",
        "Identifies information needs",
    ),
)


# Retrieve information Needed Agent
# Retrieves the necessary information for decision-making.
retrieve_information_needed_agent = Agent(
    model=_agent_model("retrieve_information_needed_agent"),
    system_prompt=load_prompt("retrieve_information_needed_agent.txt")
)
agent_registry.register(
    "retrieve_information_needed_agent",
    retrieve_information_needed_agent,
    _meta(
        "retrieve_information_needed_agent",
        "researcher",
        "Retrieves needed information",
    ),
)


# Draft Update Agent
# Updates the draft decision based on new information.
draft_update_agent = Agent(
    model=_agent_model("draft_update_agent"),
    system_prompt=load_prompt("draft_update_agent.txt")
)
agent_registry.register(
    "draft_update_agent",
    draft_update_agent,
    _meta(
        "draft_update_agent",
        "writer",
        "Updates the draft decision",
    ),
)


# Generation of Alternatives Agent
# Generates alternative options for decision-making.
generation_of_alternatives_agent = Agent(
    model=_agent_model("generation_of_alternatives_agent"),
    system_prompt=load_prompt("generation_of_alternatives_agent.txt")
)
agent_registry.register(
    "generation_of_alternatives_agent",
    generation_of_alternatives_agent,
    _meta(
        "generation_of_alternatives_agent",
        "ideation",
        "Generates alternative options",
    ),
)


# Result Agent
# Evaluates and presents the final decision outcome.
result_agent = Agent(
    model=_agent_model("result_agent"),
    output_type=ResultOutput,
    system_prompt=load_prompt("result_agent.txt")
)
agent_registry.register(
    "result_agent",
    result_agent,
    _meta(
        "result_agent",
        "decision",
        "Evaluates and presents the final decision outcome",
    ),
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
