"""

This module contains all graph nodes (agent and evaluator nodes) that define 
the workflow for the decision-making process. Each node represents a step in 
the decision-making graph and handles both the agent execution and evaluation.

Node Types:
- Agent Nodes: Execute decision-making tasks using specialized agents
- Evaluator Nodes: Validate agent outputs and control workflow branching


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from rich.prompt import Prompt

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_ai import format_as_xml

from backend.app.models.domain import DecisionState, ResultOutput
from backend.app.core.agents.decision_agents import (
    identify_trigger_agent,
    root_cause_analyzer_agent,
    scope_definition_agent,
    drafting_agent,
    establish_goals_agent,
    identify_information_needed_agent,
    retrieve_information_needed_agent,
    draft_update_agent,
    generation_of_alternatives_agent,
    result_agent
)
from backend.app.core.agents.evaluator_agents import (
    identify_trigger_agent_evaluator,
    root_cause_analyzer_agent_evaluator,
    scope_definition_agent_evaluator,
    drafting_agent_evaluator,
    establish_goals_agent_evaluator,
    identify_information_needed_agent_evaluator,
    retrieve_information_needed_agent_evaluator,
    draft_update_agent_evaluator,
    generation_of_alternatives_agent_evaluator
)

from backend.app.core.observability.tracer import trace_agent


# Instrumented agent runners to centralize tracing inputs
@trace_agent(
    "identify_trigger_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
    },
)
async def run_identify_trigger(state: DecisionState, prompt: str, *, retry_count: int = 0, has_evaluation: bool = False):
    return await identify_trigger_agent.run(prompt)


@trace_agent(
    "root_cause_analyzer_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
    },
)
async def run_root_cause_analyzer(state: DecisionState, prompt: str, *, retry_count: int = 0, has_evaluation: bool = False):
    return await root_cause_analyzer_agent.run(prompt)


@trace_agent(
    "scope_definition_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
    },
)
async def run_scope_definition(state: DecisionState, prompt: str, *, retry_count: int = 0, has_evaluation: bool = False):
    return await scope_definition_agent.run(prompt)


@trace_agent(
    "drafting_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
    },
)
async def run_drafting(state: DecisionState, prompt: str, *, retry_count: int = 0, has_evaluation: bool = False):
    return await drafting_agent.run(prompt)


@trace_agent(
    "establish_goals_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
    },
)
async def run_establish_goals(state: DecisionState, prompt: str, *, retry_count: int = 0, has_evaluation: bool = False):
    return await establish_goals_agent.run(prompt)


@trace_agent(
    "identify_information_needed_agent",
    input_extractor=lambda args, kwargs: {
        "has_evaluation": kwargs.get("has_evaluation", False),
        "has_complementary": kwargs.get("has_complementary", False),
    },
)
async def run_identify_information_needed(
    state: DecisionState,
    prompt: str,
    *,
    has_evaluation: bool = False,
    has_complementary: bool = False,
):
    return await identify_information_needed_agent.run(prompt)


@trace_agent(
    "retrieve_information_needed_agent",
    input_extractor=lambda args, kwargs: {
        "has_evaluation": kwargs.get("has_evaluation", False),
        "info_needed_length": kwargs.get("info_needed_length", 0),
    },
)
async def run_retrieve_information_needed(
    state: DecisionState,
    prompt: str,
    *,
    has_evaluation: bool = False,
    info_needed_length: int = 0,
):
    return await retrieve_information_needed_agent.run(prompt)


@trace_agent(
    "draft_update_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
        "complementary_info_num": kwargs.get("complementary_info_num", 0),
    },
)
async def run_draft_update(
    state: DecisionState,
    prompt: str,
    *,
    retry_count: int = 0,
    has_evaluation: bool = False,
    complementary_info_num: int = 0,
):
    return await draft_update_agent.run(prompt)


@trace_agent(
    "generation_of_alternatives_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
    },
)
async def run_generation_of_alternatives(
    state: DecisionState,
    prompt: str,
    *,
    retry_count: int = 0,
    has_evaluation: bool = False,
):
    return await generation_of_alternatives_agent.run(prompt)


@trace_agent(
    "result_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
    },
)
async def run_result_agent(
    state: DecisionState,
    prompt: str,
    *,
    retry_count: int = 0,
    has_evaluation: bool = False,
):
    return await result_agent.run(prompt)



# Agent Node Definitions


# Get Decision Node - Initial node to get the decision request.
@dataclass
class GetDecision(BaseNode[DecisionState]):
    """Node to get the initial decision state."""

    async def run(self, ctx: GraphRunContext[DecisionState]) -> IdentifyTrigger:
        # Only prompt if decision_requested is not already set (for API usage)
        if not ctx.state.decision_requested:
            decision_query = Prompt.ask('What is the decision you want me to help?')
            ctx.state.decision_requested = decision_query
        return IdentifyTrigger()
    

# Identify Trigger Node - Identifies the trigger for the decision.
@dataclass
class IdentifyTrigger(BaseNode[DecisionState]):
    """Node to identify the trigger for the decision."""

    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Evaluate_IdentifyTrigger:
        base_prompt = f"Here the decision requested by user: {ctx.state.decision_requested}"
        
        if self.evaluation:
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {self.evaluation}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        result = await run_identify_trigger(
            ctx.state,
            prompt,
            retry_count=self.retry_count,
            has_evaluation=bool(self.evaluation),
        )
        return Evaluate_IdentifyTrigger(answer=result.output, retry_count=self.retry_count)
    

# Analyze Root Cause Node - Analyzes the root cause of the decision trigger.
@dataclass
class AnalyzeRootCause(BaseNode[DecisionState]):
    """
    Analyzes the root cause of the identified trigger.
    Supports re-evaluation with feedback loop.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Evaluate_AnalyzeRootCause:
        base_prompt = (
            f"Here the decision requested by user: {ctx.state.decision_requested}\n"
            f"Here the identified trigger: {ctx.state.trigger}"
        )
        
        if self.evaluation:
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {self.evaluation}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        result = await run_root_cause_analyzer(
            ctx.state,
            prompt,
            retry_count=self.retry_count,
            has_evaluation=bool(self.evaluation),
        )
        return Evaluate_AnalyzeRootCause(result.output, retry_count=self.retry_count)
    

# Scope Definition Node - Defines the scope of the decision-making process.
@dataclass
class ScopeDefinition(BaseNode[DecisionState]):
    """
    Defines the scope and boundaries of the decision.
    Supports re-evaluation with feedback loop.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Evaluate_ScopeDefinition:
        base_prompt = (
            f"Here the decision requested by user: {ctx.state.decision_requested}\n"
            f"Here the identified trigger: {ctx.state.trigger}\n"
            f"Here the root cause analysis: {ctx.state.root_cause}"
        )
        
        if self.evaluation:
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {self.evaluation}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        result = await run_scope_definition(
            ctx.state,
            prompt,
            retry_count=self.retry_count,
            has_evaluation=bool(self.evaluation),
        )
        return Evaluate_ScopeDefinition(result.output, retry_count=self.retry_count)
    

# Drafting Node - Drafts the initial decision based on analysis and scope.
@dataclass
class Drafting(BaseNode[DecisionState]):
    """
    Creates initial draft of the decision based on previous analyses.
    Supports re-evaluation with feedback loop.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Evaluate_Drafting:
        base_prompt = (
            f"Here the decision requested by user: {ctx.state.decision_requested}\n"
            f"Here the identified trigger: {ctx.state.trigger}\n"
            f"Here the root cause analysis: {ctx.state.root_cause}\n"
            f"Here the scope definition: {ctx.state.scope_definition}"
        )
        
        if self.evaluation:
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {self.evaluation}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
            print("\n\n Drafting Prompt: ", prompt)
        
        result = await run_drafting(
            ctx.state,
            prompt,
            retry_count=self.retry_count,
            has_evaluation=bool(self.evaluation),
        )
        return Evaluate_Drafting(result.output, retry_count=self.retry_count)
    

# Establish Goals Node - Establishes clear goals for the decision-making process.
@dataclass
class EstablishGoals(BaseNode[DecisionState]):
    """
    Establishes SMART goals for the decision.
    Supports re-evaluation with feedback loop.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Evaluate_EstablishGoals:
        base_prompt = f"Here the decision requested by user: {ctx.state.decision_drafted}"
        
        if self.evaluation:
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {self.evaluation}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        result = await run_establish_goals(
            ctx.state,
            prompt,
            retry_count=self.retry_count,
            has_evaluation=bool(self.evaluation),
        )
        return Evaluate_EstablishGoals(result.output, retry_count=self.retry_count)
    
    
# Identify Information Needed Node - Identifies information required for decision-making.
@dataclass
class IdentifyInformationNeeded(BaseNode[DecisionState]):
    """
    Identifies additional information needed for the decision.
    Supports re-evaluation with feedback loop and complementary info iteration.
    """
    
    evaluation: Optional[str] = None
    complementary_info: Optional[bool] = None
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Evaluate_IdentifyInformationNeeded:
        base_prompt = (
            f"Here the decision requested by user: {ctx.state.decision_drafted}\n"
            f"Here the established goals for the decision: {ctx.state.goals}"
        )
        
        if self.evaluation:
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {self.evaluation}\n"
                f"Please fix it and give the correct answer."
            )
        elif self.complementary_info:
            prompt = (
                f"{base_prompt}\n"
                f"Here the complementary info about the decision: {ctx.state.complementary_info}"
            )
        else:
            prompt = base_prompt
        
        result = await run_identify_information_needed(
            ctx.state,
            prompt,
            has_evaluation=bool(self.evaluation),
            has_complementary=bool(self.complementary_info),
        )
        return Evaluate_IdentifyInformationNeeded(result.output)
    
    
# Retrieve Information Needed Node - Retrieves necessary information for decision-making.
@dataclass
class RetrieveInformationNeeded(BaseNode[DecisionState]):
    """
    Retrieves the information identified as needed.
    Supports re-evaluation with feedback loop.
    """
    
    info_needed: str
    evaluation: Optional[str] = None
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> "Evaluate_RetrieveInformationNeeded":
        base_prompt = format_as_xml({
            'decision requested': ctx.state.decision_drafted,
            'info needed': self.info_needed
        })
        
        if self.evaluation:
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {self.evaluation}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        result = await run_retrieve_information_needed(
            ctx.state,
            prompt,
            has_evaluation=bool(self.evaluation),
            info_needed_length=len(self.info_needed),
        )
        return Evaluate_RetrieveInformationNeeded(result.output)
    

# Draft Update Node - Updates the draft decision based on new information.
@dataclass
class UpdateDraft(BaseNode[DecisionState]):
    """
    Updates the decision draft with complementary information.
    Supports re-evaluation with feedback loop.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Evaluate_UpdateDraft:
        base_prompt = f"Here the decision requested by user: {ctx.state.decision_drafted}"
        
        if ctx.state.complementary_info_num > 0:
            base_prompt += f"\nHere the complementary info for the decision: {ctx.state.complementary_info}"
        
        if self.evaluation:
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {self.evaluation}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        result = await run_draft_update(
            ctx.state,
            prompt,
            retry_count=self.retry_count,
            has_evaluation=bool(self.evaluation),
            complementary_info_num=ctx.state.complementary_info_num,
        )
        return Evaluate_UpdateDraft(result.output, retry_count=self.retry_count)
    

# Generation of Alternatives Node - Generates alternative options for decision-making.
@dataclass
class GenerationOfAlternatives(BaseNode[DecisionState]):
    """
    Generates alternative options for the decision.
    Supports re-evaluation with feedback loop.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Evaluate_GenerationOfAlternatives:
        base_prompt = f"Here the decision requested by user: {ctx.state.decision_draft_updated}"
        
        if self.evaluation:
            prompt = (
                f"{base_prompt}\n"
                f"Here the current alternatives for this decision: {ctx.state.alternatives}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {self.evaluation}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        result = await run_generation_of_alternatives(
            ctx.state,
            prompt,
            retry_count=self.retry_count,
            has_evaluation=bool(self.evaluation),
        )
        return Evaluate_GenerationOfAlternatives(result.output, retry_count=self.retry_count)
    
    
# Result Node - Final node to evaluate and present the decision outcome.
@dataclass
class Result(BaseNode[DecisionState]):
    """
    Evaluates and selects the best alternative for the decision.
    Produces final decision output with commentary.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Evaluate_Result:
        base_prompt = (
            f"Here the decision requested by user: {ctx.state.decision_draft_updated}\n"
            f"Here the current alternatives for this decision: {ctx.state.alternatives}"
        )
        
        if self.evaluation:
            prompt = (
                f"{base_prompt}\n"
                f"Here the selected result for the decision: {ctx.state.result}\n"
                f"Here the comment on selected result for the decision: {ctx.state.result_comment}\n"
                f"Here the selected best alternative for the decision: {ctx.state.best_alternative_result}\n"
                f"Here the comment on selected best alternative for the decision: {ctx.state.best_alternative_result_comment}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {self.evaluation}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        result = await run_result_agent(
            ctx.state,
            prompt,
            retry_count=self.retry_count,
            has_evaluation=bool(self.evaluation),
        )
        return Evaluate_Result(result.output, retry_count=self.retry_count)
    



# Evaluator Node Definitions


# Evaluate Identify Trigger Node - Evaluates the output of Identify Trigger Node.
@dataclass
class Evaluate_IdentifyTrigger(BaseNode[DecisionState, None, str]):
    """
    Evaluates the identified trigger.
    If correct: updates state and proceeds to AnalyzeRootCause
    If incorrect: returns to IdentifyTrigger with feedback
    Max 2 retries, then accepts and proceeds.
    """
    
    answer: str
    retry_count: int = 0
    
    async def run(
        self,
        ctx: GraphRunContext[DecisionState],
    ) -> IdentifyTrigger | AnalyzeRootCause:
        assert self.answer is not None
        
        result = await identify_trigger_agent_evaluator.run(
            format_as_xml({
                'decision requested': ctx.state.decision_requested,
                'identified trigger for the decision': self.answer
            })
        )
        
        if result.output.correct or (self.retry_count >= 2):
            ctx.state.trigger = self.answer
            print("#" * 50)
            print("\n Evaluate_IdentifyTrigger")
            if result.output.correct:
                print("\n Correct Answer \n")
            else:
                print("\n Max Retries Reached - Accepting Answer \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return AnalyzeRootCause()
        else:
            print("#" * 50)
            print("\n Evaluate_IdentifyTrigger")
            print("\n Wrong Answer - Retrying \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return IdentifyTrigger(evaluation=result.output.comment, retry_count=self.retry_count + 1)
        
        
# Evaluate Analyze Root Cause Node - Evaluates the output of Analyze Root Cause Node.
@dataclass
class Evaluate_AnalyzeRootCause(BaseNode[DecisionState, None, str]):
    """
    Evaluates the root cause analysis.
    If correct: updates state and proceeds to ScopeDefinition
    If incorrect: returns to AnalyzeRootCause with feedback
    Max 2 retries, then accepts and proceeds.
    """
    
    answer: str
    retry_count: int = 0
    
    async def run(
        self,
        ctx: GraphRunContext[DecisionState],
    ) -> AnalyzeRootCause | ScopeDefinition:
        assert self.answer is not None
        
        result = await root_cause_analyzer_agent_evaluator.run(
            format_as_xml({
                'decision requested': ctx.state.decision_requested,
                'identified trigger for the decision': ctx.state.trigger,
                'root cause analysis': self.answer
            })
        )
        
        if result.output.correct or (self.retry_count >= 2):
            ctx.state.root_cause = self.answer
            print("#" * 50)
            print("\n Evaluate_AnalyzeRootCause")
            if result.output.correct:
                print("\n Correct Answer \n")
            else:
                print("\n Max Retries Reached - Accepting Answer \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return ScopeDefinition()
        else:
            print("#" * 50)
            print("\n Evaluate_AnalyzeRootCause")
            print("\n Wrong Answer - Retrying \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return AnalyzeRootCause(evaluation=result.output.comment, retry_count=self.retry_count + 1)


# Evaluate Scope Definition Node - Evaluates the output of Scope Definition Node.
@dataclass
class Evaluate_ScopeDefinition(BaseNode[DecisionState, None, str]):
    """
    Evaluates the scope definition.
    If correct: updates state and proceeds to Drafting
    If incorrect: returns to ScopeDefinition with feedback
    Max 2 retries, then accepts and proceeds.
    """
    
    answer: str
    retry_count: int = 0
    
    async def run(
        self,
        ctx: GraphRunContext[DecisionState],
    ) -> ScopeDefinition | Drafting:
        assert self.answer is not None
        
        result = await scope_definition_agent_evaluator.run(
            format_as_xml({
                'decision requested': ctx.state.decision_requested,
                'identified trigger for the decision': ctx.state.trigger,
                'root cause analysis': ctx.state.root_cause,
                'scope definition': self.answer
            })
        )
        
        if result.output.correct or (self.retry_count >= 2):
            ctx.state.scope_definition = self.answer
            print("#" * 50)
            print("\n Evaluate_ScopeDefinition")
            if result.output.correct:
                print("\n Correct Answer \n")
            else:
                print("\n Max Retries Reached - Accepting Answer \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return Drafting()
        else:
            print("#" * 50)
            print("\n Evaluate_ScopeDefinition")
            print("\n Wrong Answer - Retrying \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return ScopeDefinition(evaluation=result.output.comment, retry_count=self.retry_count + 1)


# Evaluate Drafting Node - Evaluates the output of Drafting Node.
@dataclass
class Evaluate_Drafting(BaseNode[DecisionState, None, str]):
    """
    Evaluates the decision draft.
    If correct: updates state and proceeds to EstablishGoals
    If incorrect: returns to Drafting with feedback
    Max 2 retries, then accepts and proceeds.
    """
    
    answer: str
    retry_count: int = 0
    
    async def run(
        self,
        ctx: GraphRunContext[DecisionState],
    ) -> Drafting | EstablishGoals:
        assert self.answer is not None
        
        result = await drafting_agent_evaluator.run(
            format_as_xml({
                'decision requested': ctx.state.decision_requested,
                'identified trigger for the decision': ctx.state.trigger,
                'root cause analysis': ctx.state.root_cause,
                'scope definition': ctx.state.scope_definition,
                'decision drafted': self.answer
            })
        )
        
        if result.output.correct or (self.retry_count >= 2):
            ctx.state.decision_drafted = self.answer
            print("#" * 50)
            print("\n Evaluate_Drafting")
            if result.output.correct:
                print("\n Correct Answer \n")
            else:
                print("\n Max Retries Reached - Accepting Answer \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return EstablishGoals()
        else:
            print("#" * 50)
            print("\n Evaluate_Drafting")
            print("\n Wrong Answer - Retrying \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return Drafting(evaluation=result.output.comment, retry_count=self.retry_count + 1)


# Evaluate Establish Goals Node - Evaluates the output of Establish Goals Node.
@dataclass
class Evaluate_EstablishGoals(BaseNode[DecisionState, None, str]):
    """
    Evaluates the established goals.
    If correct: updates state and proceeds to IdentifyInformationNeeded
    If incorrect: returns to EstablishGoals with feedback
    Max 2 retries, then accepts and proceeds.
    """
    
    answer: str
    retry_count: int = 0
    
    async def run(
        self,
        ctx: GraphRunContext[DecisionState],
    ) -> EstablishGoals | IdentifyInformationNeeded:
        assert self.answer is not None
        
        result = await establish_goals_agent_evaluator.run(
            format_as_xml({
                'decision requested': ctx.state.decision_drafted
            })
        )
        
        if result.output.correct or (self.retry_count >= 2):
            ctx.state.goals = self.answer
            print("#" * 50)
            print("\n Evaluate_EstablishGoals")
            if result.output.correct:
                print("\n Correct Answer \n")
            else:
                print("\n Max Retries Reached - Accepting Answer \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return IdentifyInformationNeeded()
        else:
            print("#" * 50)
            print("\n Evaluate_EstablishGoals")
            print("\n Wrong Answer - Retrying \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return EstablishGoals(evaluation=result.output.comment, retry_count=self.retry_count + 1)


# Evaluate Identify Information Needed Node - Evaluates the output of Identify Information Needed Node.
@dataclass
class Evaluate_IdentifyInformationNeeded(BaseNode[DecisionState, None, str]):
    """
    Evaluates the identified information needs.
    If correct OR max iterations (3): proceeds to UpdateDraft
    If incorrect: retrieves info and returns to RetrieveInformationNeeded
    """
    
    answer: str
    
    async def run(
        self,
        ctx: GraphRunContext[DecisionState],
    ) -> Union["RetrieveInformationNeeded", UpdateDraft]:
        assert self.answer is not None
        
        result = await identify_information_needed_agent_evaluator.run(
            format_as_xml({
                'decision requested': ctx.state.decision_drafted
            })
        )
        
        if result.output.correct or (ctx.state.complementary_info_num >= 3):
            print("#" * 50)
            print("\n Evaluate_IdentifyInformationNeeded")
            print("\n Correct Answer \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print("#" * 50 + "\n")
            return UpdateDraft()
        else:
            # Information needs identified, proceed to retrieve it
            print("#" * 50)
            print("\n Evaluate_IdentifyInformationNeeded")
            print("\n Proceeding to Information Retrieval \n")
            print("#" * 50 + "\n")
            print(f"\nInfo Needed: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print("#" * 50 + "\n")
            return RetrieveInformationNeeded(info_needed=self.answer)
        

# Evaluate Retrieve Information Needed Node - Evaluates the output of Retrieve Information Needed Node.
@dataclass
class Evaluate_RetrieveInformationNeeded(BaseNode[DecisionState, None, str]):
    """
    Evaluates the retrieved information quality.
    If correct: adds to complementary_info and returns to IdentifyInformationNeeded
    If incorrect: returns to RetrieveInformationNeeded with feedback
    Max 2 retries, then accepts and proceeds.
    """
    
    answer: str
    retry_count: int = 0
    
    async def run(
        self,
        ctx: GraphRunContext[DecisionState],
    ) -> Union[RetrieveInformationNeeded, IdentifyInformationNeeded]:
        assert self.answer is not None
        
        result = await retrieve_information_needed_agent_evaluator.run(
            format_as_xml({
                'decision requested': ctx.state.decision_drafted,
                'info needed': ctx.state.complementary_info if ctx.state.complementary_info else "Not specified",
                'retrieved information': self.answer
            })
        )
        
        if result.output.correct or (self.retry_count >= 2):
            # Information quality approved or max retries reached
            ctx.state.complementary_info += "\n" + self.answer
            ctx.state.complementary_info_num += 1
            print("#" * 50)
            print("\n Evaluate_RetrieveInformationNeeded")
            print("\n Information Approved \n")
            print("#" * 50 + "\n")
            print(f"\nRetrieved Info: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print("#" * 50 + "\n")
            return IdentifyInformationNeeded(complementary_info=True)
        else:
            # Information needs improvement
            print("#" * 50)
            print("\n Evaluate_RetrieveInformationNeeded")
            print("\n Revision Required \n")
            print("#" * 50 + "\n")
            print(f"\nRetrieved Info: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print("#" * 50 + "\n")
            # Get the info_needed from the previous RetrieveInformationNeeded call
            # We need to pass it through - for now, extract from context or make assumption
            info_needed = "Information as previously identified"  # TODO: Better state management
            return RetrieveInformationNeeded(
                info_needed=info_needed,
                evaluation=result.output.comment
            )

    
# Evaluate Draft Update Node - Evaluates the output of Draft Update Node.
@dataclass
class Evaluate_UpdateDraft(BaseNode[DecisionState, None, str]):
    """
    Evaluates the updated draft.
    If correct: updates state and proceeds to GenerationOfAlternatives
    If incorrect: returns to UpdateDraft with feedback
    Max 2 retries, then accepts and proceeds.
    """
    
    answer: str
    retry_count: int = 0
    
    async def run(
        self,
        ctx: GraphRunContext[DecisionState],
    ) -> UpdateDraft | GenerationOfAlternatives:
        assert self.answer is not None
        
        result = await draft_update_agent_evaluator.run(
            format_as_xml({
                'decision requested': ctx.state.decision_drafted
            })
        )
        
        if result.output.correct or (self.retry_count >= 2):
            ctx.state.decision_draft_updated = self.answer
            print("#" * 50)
            print("\n Evaluate_UpdateDraft")
            if result.output.correct:
                print("\n Correct Answer \n")
            else:
                print("\n Max Retries Reached - Accepting Answer \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return GenerationOfAlternatives()
        else:
            print("#" * 50)
            print("\n Evaluate_UpdateDraft")
            print("\n Wrong Answer - Retrying \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return UpdateDraft(evaluation=result.output.comment, retry_count=self.retry_count + 1)
        
        
# Evaluate Generation of Alternatives Node - Evaluates the output of Generation of Alternatives Node.
@dataclass
class Evaluate_GenerationOfAlternatives(BaseNode[DecisionState, None, str]):
    """
    Evaluates the generated alternatives.
    If correct: updates state and proceeds to Result
    If incorrect: returns to GenerationOfAlternatives with feedback
    Max 2 retries, then accepts and proceeds.
    """
    
    answer: str
    retry_count: int = 0
    
    async def run(
        self,
        ctx: GraphRunContext[DecisionState],
    ) -> GenerationOfAlternatives | Result:
        assert self.answer is not None
        
        result = await generation_of_alternatives_agent_evaluator.run(
            format_as_xml({
                'decision requested': ctx.state.decision_drafted
            })
        )
        
        if result.output.correct or (self.retry_count >= 2):
            ctx.state.alternatives = self.answer
            print("#" * 50)
            print("\n Evaluate_GenerationOfAlternatives")
            if result.output.correct:
                print("\n Correct Answer \n")
            else:
                print("\n Max Retries Reached - Accepting Answer \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return Result()
        else:
            print("#" * 50)
            print("\n Evaluate_GenerationOfAlternatives")
            print("\n Wrong Answer - Retrying \n")
            print("#" * 50 + "\n")
            print(f"\nAnswer: {self.answer}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return GenerationOfAlternatives(evaluation=result.output.comment, retry_count=self.retry_count + 1)
        

# Evaluate Result Node - Evaluates the output of Result Node.
@dataclass
class Evaluate_Result(BaseNode[DecisionState, None, str]):
    """
    Evaluates the final decision result.
    If correct: updates state and ends the workflow
    If incorrect: returns to Result with feedback
    Max 2 retries, then accepts and proceeds.
    """
    
    answer: ResultOutput
    retry_count: int = 0
    
    async def run(
        self,
        ctx: GraphRunContext[DecisionState],
    ) -> Result | End:
        assert self.answer is not None
        
        result = await draft_update_agent_evaluator.run(
            format_as_xml({
                'decision requested': ctx.state.decision_drafted
            })
        )
        
        if result.output.correct or (self.retry_count >= 2):
            ctx.state.result = self.answer.result
            ctx.state.result_comment = self.answer.result_comment
            ctx.state.best_alternative_result = self.answer.best_alternative_result
            ctx.state.best_alternative_result_comment = self.answer.best_alternative_result_comment
            if result.output.correct:
                print("#" * 50)
                print("\n Evaluate_Result")
                print("\n Correct Answer \n")
                print("#" * 50 + "\n")
            else:
                print("#" * 50)
                print("\n Evaluate_Result")
                print("\n Max Retries Reached - Accepting Answer \n")
                print("#" * 50 + "\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return End(True)
        else:
            print("#" * 50)
            print("\n Evaluate_Result")
            print("\n Wrong Answer - Retrying \n")
            print("#" * 50 + "\n")
            print(f"\nSelected Decision: {self.answer.result}\n\n")
            print(f"\nSelected Decision Comment: {self.answer.result_comment}\n\n")
            print(f"\nAlternative Decision: {self.answer.best_alternative_result}\n\n")
            print(f"\nAlternative Decision Comment: {self.answer.best_alternative_result_comment}\n\n")
            print(f"\nEvaluation: {result.output.comment}\n")
            print(f"\nRetry count: {self.retry_count}\n")
            print("#" * 50 + "\n")
            return Result(evaluation=result.output.comment, retry_count=self.retry_count + 1)
