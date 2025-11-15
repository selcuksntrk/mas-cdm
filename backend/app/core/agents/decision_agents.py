"""

Decision Agents Module - This module contains classes and functions related 
to decision-making agents within the MAS-CDM backend application.

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
