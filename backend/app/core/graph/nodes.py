"""

This module contains all graph nodes (agent and evaluator nodes) that define 
the workflow for the decision-making process. Each node represents a step in 
the decision-making graph and handles both the agent execution and evaluation.

Node Types:
- Agent Nodes: Execute decision-making tasks using specialized agents
- Evaluator Nodes: Validate agent outputs and control workflow branching


"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Union

# Note: rich.prompt.Prompt is imported but no longer used for blocking input
# Decision requests must be provided via API or state initialization

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_ai import format_as_xml

from backend.app.config import get_settings
from backend.app.models.domain import DecisionState, ResultOutput, EvaluationOutput
from backend.app.utils.helpers import truncate_text
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
    generation_of_alternatives_agent_evaluator,
    result_agent_evaluator
)

from backend.app.core.observability.tracer import trace_agent
from backend.app.core.memory.store import memory_store
from backend.app.core.resilience.retry import CircuitBreaker, RetryPolicy, execute_with_resilience
from backend.app.core.agents.lifecycle import lifecycle_manager
from backend.app.core.exceptions import (
    LLMProviderError,
    LLMAPIConnectionError,
    LLMRateLimitError,
    RetryError,
    CircuitBreakerOpenError,
)


logger = logging.getLogger(__name__)
settings = get_settings()


def _log_evaluation_result(
    node_name: str,
    status: str,
    answer: str,
    evaluation_comment: str,
    retry_count: int,
    info_field_name: str = "Answer"
) -> None:
    """
    Structured logging for evaluation node results.
    
    Args:
        node_name: Name of the evaluation node
        status: Result status (e.g., "correct", "max_retries", "retry")
        answer: The agent's answer
        evaluation_comment: Evaluator's feedback
        retry_count: Current retry iteration
        info_field_name: Label for the answer field (default "Answer")
    """
    logger.debug(
        "Evaluation completed",
        extra={
            "node": node_name,
            "status": status,
            "retry_count": retry_count,
            info_field_name.lower(): answer[:200] + "..." if len(answer) > 200 else answer,
            "evaluation": evaluation_comment,
        }
    )


def _log_agent_start(node_name: str, has_evaluation: bool = False, retry_count: int = 0) -> None:
    """Log when an agent node starts execution."""
    logger.debug(
        "Agent starting",
        extra={
            "node": node_name,
            "is_retry": has_evaluation,
            "retry_count": retry_count,
        }
    )


def _log_prompt_debug(node_name: str, prompt: str) -> None:
    """Log prompt content for debugging purposes."""
    logger.debug(
        "Agent prompt",
        extra={
            "node": node_name,
            "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
        }
    )


def _safe_eval_output(result, default_correct: bool = False) -> tuple[bool, str]:
    """
    Safely extract evaluation output with defensive checks.
    
    Handles cases where the LLM might return unexpected output structure
    or when PydanticAI fails to parse the response properly.
    
    Args:
        result: The result from evaluator agent.run()
        default_correct: Default value for 'correct' if extraction fails
        
    Returns:
        Tuple of (is_correct: bool, comment: str)
    """
    try:
        output = result.output
        
        # Handle case where output might be None
        if output is None:
            logger.warning("Evaluator returned None output, using defaults")
            return default_correct, "Evaluation failed - no output returned"
        
        # Handle EvaluationOutput model
        if isinstance(output, EvaluationOutput):
            return output.correct, output.comment or "No comment provided"
        
        # Handle dict-like output (shouldn't happen with output_type but defensive)
        if hasattr(output, 'get'):
            return output.get('correct', default_correct), output.get('comment', 'No comment')
        
        # Handle object with attributes
        correct = getattr(output, 'correct', default_correct)
        comment = getattr(output, 'comment', 'No comment provided')
        
        # Validate types
        if not isinstance(correct, bool):
            logger.warning(f"Evaluator 'correct' field is not bool: {type(correct)}")
            correct = bool(correct) if correct is not None else default_correct
        
        if not isinstance(comment, str):
            comment = str(comment) if comment is not None else "No comment provided"
        
        return correct, comment
        
    except Exception as e:
        logger.error(f"Failed to extract evaluator output: {e}", exc_info=True)
        return default_correct, f"Evaluation extraction failed: {str(e)}"


# Shared retry/circuit configuration for agent calls
AGENT_RETRY_POLICY = RetryPolicy(
    max_attempts=settings.agent_max_retries,
    backoff_factor=settings.agent_retry_backoff,
    max_backoff=settings.agent_retry_max_backoff,
    jitter=settings.agent_retry_jitter,
    retryable_exceptions=(
        LLMProviderError,
        LLMAPIConnectionError,
        LLMRateLimitError,
        TimeoutError,
    ),
)

AGENT_CIRCUIT_BREAKER = CircuitBreaker(
    failure_threshold=settings.circuit_breaker_failure_threshold,
    recovery_time=settings.circuit_breaker_recovery_time,
    half_open_success_threshold=settings.circuit_breaker_half_open_success_threshold,
)


async def _run_agent_with_model(agent, prompt: str, model_name: str):
    """Execute an agent with a specific model, restoring the original afterwards."""
    original_model = getattr(agent, "model", None)
    try:
        if hasattr(agent, "model"):
            setattr(agent, "model", model_name)
        return await agent.run(prompt)
    finally:
        if hasattr(agent, "model"):
            setattr(agent, "model", original_model)


async def _execute_agent(agent_name: str, agent, prompt: str):
    """
    Centralized agent execution with lifecycle management, retries, fallback model, and circuit breaker.
    
    This function:
    1. Enforces lifecycle limits (max concurrency, timeouts)
    2. Executes agent with retry policy and circuit breaker
    3. Falls back to alternative model if configured
    4. Properly cleans up resources via lifecycle manager
    
    Raises:
        RetryError: When all retry attempts are exhausted
        CircuitBreakerOpenError: When circuit breaker is open
    """
    
    # Step 1: Initialize agent via lifecycle manager (enforces concurrency & timeout)
    try:
        await lifecycle_manager.initialize_agent(agent_name)
    except Exception as e:
        logger.error(f"Failed to initialize agent {agent_name}: {e}")
        raise
    
    try:
        # Step 2: Define the primary agent call with timeout
        async def primary_call():
            return await asyncio.wait_for(
                agent.run(prompt),
                timeout=settings.agent_call_timeout
            )

        # Step 3: Define fallback call if configured
        fallback_call = None
        if settings.fallback_model_name:
            async def fallback_call():
                return await asyncio.wait_for(
                    _run_agent_with_model(agent, prompt, settings.fallback_model_name),
                    timeout=settings.agent_call_timeout
                )

        # Step 4: Define retry callback
        def on_retry(attempt: int, error: BaseException) -> None:
            logger.warning(
                "Agent %s retry %s/%s after error: %s",
                agent_name,
                attempt,
                AGENT_RETRY_POLICY.max_attempts,
                error,
            )

        # Step 5: Execute with resilience (retries + circuit breaker)
        return await execute_with_resilience(
            primary_call,
            fallback=fallback_call,
            policy=AGENT_RETRY_POLICY,
            circuit_breaker=AGENT_CIRCUIT_BREAKER,
            on_retry=on_retry,
        )
    
    finally:
        # Step 6: Always terminate agent to free resources
        try:
            await lifecycle_manager.terminate_agent(agent_name)
        except Exception as e:
            logger.error(f"Failed to terminate agent {agent_name}: {e}")


def _create_error_state_update(node_name: str, error: Exception) -> str:
    """
    Create a structured error message for state updates when agent execution fails.
    
    Args:
        node_name: Name of the node that encountered the error
        error: The exception that was raised
        
    Returns:
        Formatted error message for inclusion in state
    """
    error_type = type(error).__name__
    if isinstance(error, RetryError):
        return f"[ERROR] {node_name} failed after exhausting all retry attempts. Error: {str(error)}"
    elif isinstance(error, CircuitBreakerOpenError):
        return f"[ERROR] {node_name} circuit breaker is open. The service needs time to recover. Error: {str(error)}"
    else:
        return f"[ERROR] {node_name} encountered an unexpected error ({error_type}): {str(error)}"


# Instrumented agent runners to centralize tracing inputs
@trace_agent(
    "identify_trigger_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
    },
)
async def run_identify_trigger(state: DecisionState, prompt: str, *, retry_count: int = 0, has_evaluation: bool = False):
    return await _execute_agent("identify_trigger_agent", identify_trigger_agent, prompt)


@trace_agent(
    "root_cause_analyzer_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
    },
)
async def run_root_cause_analyzer(state: DecisionState, prompt: str, *, retry_count: int = 0, has_evaluation: bool = False):
    return await _execute_agent("root_cause_analyzer_agent", root_cause_analyzer_agent, prompt)


@trace_agent(
    "scope_definition_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
    },
)
async def run_scope_definition(state: DecisionState, prompt: str, *, retry_count: int = 0, has_evaluation: bool = False):
    return await _execute_agent("scope_definition_agent", scope_definition_agent, prompt)


@trace_agent(
    "drafting_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
    },
)
async def run_drafting(state: DecisionState, prompt: str, *, retry_count: int = 0, has_evaluation: bool = False):
    return await _execute_agent("drafting_agent", drafting_agent, prompt)


@trace_agent(
    "establish_goals_agent",
    input_extractor=lambda args, kwargs: {
        "retry_count": kwargs.get("retry_count", 0),
        "has_evaluation": kwargs.get("has_evaluation", False),
    },
)
async def run_establish_goals(state: DecisionState, prompt: str, *, retry_count: int = 0, has_evaluation: bool = False):
    return await _execute_agent("establish_goals_agent", establish_goals_agent, prompt)


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
    return await _execute_agent("identify_information_needed_agent", identify_information_needed_agent, prompt)


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
    return await _execute_agent("retrieve_information_needed_agent", retrieve_information_needed_agent, prompt)


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
    return await _execute_agent("draft_update_agent", draft_update_agent, prompt)


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
    return await _execute_agent("generation_of_alternatives_agent", generation_of_alternatives_agent, prompt)


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
    return await _execute_agent("result_agent", result_agent, prompt)



# Agent Node Definitions


# Get Decision Node - Initial node to get the decision request.
@dataclass
class GetDecision(BaseNode[DecisionState]):
    """Node to get the initial decision state."""

    async def run(self, ctx: GraphRunContext[DecisionState]) -> IdentifyTrigger:
        # Validate that decision_requested is provided (required for API usage)
        if not ctx.state.decision_requested:
            raise ValueError(
                "decision_requested must be provided in state. "
                "Use the API endpoint or provide the decision query when initializing DecisionState."
            )
        # Pull memory context for the decision
        retrieved = memory_store.search(ctx.state.decision_requested, top_k=3)
        if retrieved:
            ctx.state.memory_hits = len(retrieved)
            ctx.state.memory_context = "\n".join(doc.content for doc, _ in retrieved)
        return IdentifyTrigger()
    

# Identify Trigger Node - Identifies the trigger for the decision.
@dataclass
class IdentifyTrigger(BaseNode[DecisionState]):
    """Node to identify the trigger for the decision."""

    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Union[Evaluate_IdentifyTrigger, End]:
        _log_agent_start("IdentifyTrigger", bool(self.evaluation), self.retry_count)
        
        # Truncate inputs to prevent context overflow
        decision_requested = truncate_text(ctx.state.decision_requested, max_chars=2000)
        memory_context = truncate_text(ctx.state.memory_context, max_chars=2000)
        
        base_prompt = f"Here the decision requested by user: {decision_requested}"
        if memory_context:
            base_prompt += f"\nHere is relevant context from memory: {memory_context}"
        
        if self.evaluation:
            evaluation_truncated = truncate_text(self.evaluation, max_chars=1000)
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {evaluation_truncated}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        _log_prompt_debug("IdentifyTrigger", prompt)
        
        try:
            result = await run_identify_trigger(
                ctx.state,
                prompt,
                retry_count=self.retry_count,
                has_evaluation=bool(self.evaluation),
            )
            return Evaluate_IdentifyTrigger(answer=result.output, retry_count=self.retry_count)
        
        except (RetryError, CircuitBreakerOpenError) as e:
            # Critical: Agent failed after retries or circuit breaker opened
            error_msg = _create_error_state_update("IdentifyTrigger", e)
            logger.error(error_msg, exc_info=e)
            
            # Update state with error for visibility
            ctx.state.trigger = error_msg
            
            # End the graph execution gracefully with error state
            # In production, you might want to route to HITL instead
            return End(f"Agent execution failed: {str(e)}")
    

# Analyze Root Cause Node - Analyzes the root cause of the decision trigger.
@dataclass
class AnalyzeRootCause(BaseNode[DecisionState]):
    """
    Analyzes the root cause of the identified trigger.
    Supports re-evaluation with feedback loop.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Union[Evaluate_AnalyzeRootCause, End]:
        _log_agent_start("AnalyzeRootCause", bool(self.evaluation), self.retry_count)
        
        # Truncate inputs to prevent context overflow
        decision_requested = truncate_text(ctx.state.decision_requested, max_chars=2000)
        trigger = truncate_text(ctx.state.trigger, max_chars=2000)
        memory_context = truncate_text(ctx.state.memory_context, max_chars=2000)
        
        base_prompt = (
            f"Here the decision requested by user: {decision_requested}\n"
            f"Here the identified trigger: {trigger}"
        )
        if memory_context:
            base_prompt += f"\nHere is relevant context from memory: {memory_context}"
        
        if self.evaluation:
            evaluation_truncated = truncate_text(self.evaluation, max_chars=1000)
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {evaluation_truncated}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        _log_prompt_debug("AnalyzeRootCause", prompt)
        
        try:
            result = await run_root_cause_analyzer(
                ctx.state,
                prompt,
                retry_count=self.retry_count,
                has_evaluation=bool(self.evaluation),
            )
            return Evaluate_AnalyzeRootCause(result.output, retry_count=self.retry_count)
        
        except (RetryError, CircuitBreakerOpenError) as e:
            error_msg = _create_error_state_update("AnalyzeRootCause", e)
            logger.error(error_msg, exc_info=e)
            ctx.state.root_cause = error_msg
            return End(f"Agent execution failed: {str(e)}")
    

# Scope Definition Node - Defines the scope of the decision-making process.
@dataclass
class ScopeDefinition(BaseNode[DecisionState]):
    """
    Defines the scope and boundaries of the decision.
    Supports re-evaluation with feedback loop.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Union[Evaluate_ScopeDefinition, End]:
        _log_agent_start("ScopeDefinition", bool(self.evaluation), self.retry_count)
        
        # Truncate inputs to prevent context overflow
        decision_requested = truncate_text(ctx.state.decision_requested, max_chars=2000)
        trigger = truncate_text(ctx.state.trigger, max_chars=2000)
        root_cause = truncate_text(ctx.state.root_cause, max_chars=2000)
        memory_context = truncate_text(ctx.state.memory_context, max_chars=2000)
        
        base_prompt = (
            f"Here the decision requested by user: {decision_requested}\n"
            f"Here the identified trigger: {trigger}\n"
            f"Here the root cause analysis: {root_cause}"
        )
        if memory_context:
            base_prompt += f"\nHere is relevant context from memory: {memory_context}"
        
        if self.evaluation:
            evaluation_truncated = truncate_text(self.evaluation, max_chars=1000)
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {evaluation_truncated}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        _log_prompt_debug("ScopeDefinition", prompt)
        
        try:
            result = await run_scope_definition(
                ctx.state,
                prompt,
                retry_count=self.retry_count,
                has_evaluation=bool(self.evaluation),
            )
            return Evaluate_ScopeDefinition(result.output, retry_count=self.retry_count)
        
        except (RetryError, CircuitBreakerOpenError) as e:
            error_msg = _create_error_state_update("ScopeDefinition", e)
            logger.error(error_msg, exc_info=e)
            ctx.state.scope_definition = error_msg
            return End(f"Agent execution failed: {str(e)}")
    

# Drafting Node - Drafts the initial decision based on analysis and scope.
@dataclass
class Drafting(BaseNode[DecisionState]):
    """
    Creates initial draft of the decision based on previous analyses.
    Supports re-evaluation with feedback loop.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Union[Evaluate_Drafting, End]:
        _log_agent_start("Drafting", bool(self.evaluation), self.retry_count)
        
        # Truncate inputs to prevent context overflow
        decision_requested = truncate_text(ctx.state.decision_requested, max_chars=2000)
        trigger = truncate_text(ctx.state.trigger, max_chars=2000)
        root_cause = truncate_text(ctx.state.root_cause, max_chars=2000)
        scope_definition = truncate_text(ctx.state.scope_definition, max_chars=2000)
        memory_context = truncate_text(ctx.state.memory_context, max_chars=2000)
        
        base_prompt = (
            f"Here the decision requested by user: {decision_requested}\n"
            f"Here the identified trigger: {trigger}\n"
            f"Here the root cause analysis: {root_cause}\n"
            f"Here the scope definition: {scope_definition}"
        )
        if memory_context:
            base_prompt += f"\nHere is relevant context from memory: {memory_context}"
        
        if self.evaluation:
            evaluation_truncated = truncate_text(self.evaluation, max_chars=1000)
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {evaluation_truncated}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        _log_prompt_debug("Drafting", prompt)
        
        try:
            result = await run_drafting(
                ctx.state,
                prompt,
                retry_count=self.retry_count,
                has_evaluation=bool(self.evaluation),
            )
            return Evaluate_Drafting(result.output, retry_count=self.retry_count)
        
        except (RetryError, CircuitBreakerOpenError) as e:
            error_msg = _create_error_state_update("Drafting", e)
            logger.error(error_msg, exc_info=e)
            ctx.state.decision_drafted = error_msg
            return End(f"Agent execution failed: {str(e)}")
    

# Establish Goals Node - Establishes clear goals for the decision-making process.
@dataclass
class EstablishGoals(BaseNode[DecisionState]):
    """
    Establishes SMART goals for the decision.
    Supports re-evaluation with feedback loop.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Union[Evaluate_EstablishGoals, End]:
        _log_agent_start("EstablishGoals", bool(self.evaluation), self.retry_count)
        
        # Truncate inputs to prevent context overflow
        decision_drafted = truncate_text(ctx.state.decision_drafted, max_chars=2000)
        memory_context = truncate_text(ctx.state.memory_context, max_chars=2000)
        
        base_prompt = f"Here the decision requested by user: {decision_drafted}"
        if memory_context:
            base_prompt += f"\nHere is relevant context from memory: {memory_context}"
        
        if self.evaluation:
            evaluation_truncated = truncate_text(self.evaluation, max_chars=1000)
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {evaluation_truncated}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        _log_prompt_debug("EstablishGoals", prompt)
        
        try:
            result = await run_establish_goals(
                ctx.state,
                prompt,
                retry_count=self.retry_count,
                has_evaluation=bool(self.evaluation),
            )
            return Evaluate_EstablishGoals(result.output, retry_count=self.retry_count)
        
        except (RetryError, CircuitBreakerOpenError) as e:
            error_msg = _create_error_state_update("EstablishGoals", e)
            logger.error(error_msg, exc_info=e)
            ctx.state.goals = error_msg
            return End(f"Agent execution failed: {str(e)}")
    
    
# Identify Information Needed Node - Identifies information required for decision-making.
@dataclass
class IdentifyInformationNeeded(BaseNode[DecisionState]):
    """
    Identifies additional information needed for the decision.
    Supports re-evaluation with feedback loop and complementary info iteration.
    """
    
    evaluation: Optional[str] = None
    complementary_info: Optional[bool] = None
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Union[Evaluate_IdentifyInformationNeeded, End]:
        _log_agent_start("IdentifyInformationNeeded", bool(self.evaluation), 0)
        
        # Truncate inputs to prevent context overflow
        decision_drafted = truncate_text(ctx.state.decision_drafted, max_chars=2000)
        goals = truncate_text(ctx.state.goals, max_chars=2000)
        memory_context = truncate_text(ctx.state.memory_context, max_chars=2000)
        complementary_info_text = truncate_text(ctx.state.complementary_info, max_chars=2000)
        
        base_prompt = (
            f"Here the decision requested by user: {decision_drafted}\n"
            f"Here the established goals for the decision: {goals}"
        )
        if memory_context:
            base_prompt += f"\nHere is relevant context from memory: {memory_context}"
        
        if self.evaluation:
            evaluation_truncated = truncate_text(self.evaluation, max_chars=1000)
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {evaluation_truncated}\n"
                f"Please fix it and give the correct answer."
            )
        elif self.complementary_info:
            prompt = (
                f"{base_prompt}\n"
                f"Here the complementary info about the decision: {complementary_info_text}"
            )
        else:
            prompt = base_prompt
        
        _log_prompt_debug("IdentifyInformationNeeded", prompt)
        
        try:
            result = await run_identify_information_needed(
                ctx.state,
                prompt,
                has_evaluation=bool(self.evaluation),
                has_complementary=bool(self.complementary_info),
            )
            return Evaluate_IdentifyInformationNeeded(result.output)
        
        except (RetryError, CircuitBreakerOpenError) as e:
            error_msg = _create_error_state_update("IdentifyInformationNeeded", e)
            logger.error(error_msg, exc_info=e)
            ctx.state.info_needed = error_msg
            return End(f"Agent execution failed: {str(e)}")
    
    
# Retrieve Information Needed Node - Retrieves necessary information for decision-making.
@dataclass
class RetrieveInformationNeeded(BaseNode[DecisionState]):
    """
    Retrieves the information identified as needed.
    Supports re-evaluation with feedback loop.
    """
    
    info_needed: str
    evaluation: Optional[str] = None
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Union["Evaluate_RetrieveInformationNeeded", End]:
        _log_agent_start("RetrieveInformationNeeded", bool(self.evaluation), 0)
        
        # Truncate inputs to prevent context overflow
        decision_drafted = truncate_text(ctx.state.decision_drafted, max_chars=2000)
        info_needed = truncate_text(self.info_needed, max_chars=2000)
        
        base_prompt = format_as_xml({
            'decision requested': decision_drafted,
            'info needed': info_needed
        })
        
        if self.evaluation:
            evaluation_truncated = truncate_text(self.evaluation, max_chars=1000)
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {evaluation_truncated}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        _log_prompt_debug("RetrieveInformationNeeded", prompt)
        
        try:
            result = await run_retrieve_information_needed(
                ctx.state,
                prompt,
                has_evaluation=bool(self.evaluation),
                info_needed_length=len(self.info_needed),
            )
            return Evaluate_RetrieveInformationNeeded(result.output)
        
        except (RetryError, CircuitBreakerOpenError) as e:
            error_msg = _create_error_state_update("RetrieveInformationNeeded", e)
            logger.error(error_msg, exc_info=e)
            ctx.state.complementary_info = error_msg
            return End(f"Agent execution failed: {str(e)}")
    

# Draft Update Node - Updates the draft decision based on new information.
@dataclass
class UpdateDraft(BaseNode[DecisionState]):
    """
    Updates the decision draft with complementary information.
    Supports re-evaluation with feedback loop.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Union[Evaluate_UpdateDraft, End]:
        _log_agent_start("UpdateDraft", bool(self.evaluation), self.retry_count)
        
        # Truncate inputs to prevent context overflow
        decision_drafted = truncate_text(ctx.state.decision_drafted, max_chars=2000)
        complementary_info = truncate_text(ctx.state.complementary_info, max_chars=2000)
        
        base_prompt = f"Here the decision requested by user: {decision_drafted}"
        
        if ctx.state.complementary_info_num > 0:
            base_prompt += f"\nHere the complementary info for the decision: {complementary_info}"
        
        if self.evaluation:
            evaluation_truncated = truncate_text(self.evaluation, max_chars=1000)
            prompt = (
                f"{base_prompt}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {evaluation_truncated}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        _log_prompt_debug("UpdateDraft", prompt)
        
        try:
            result = await run_draft_update(
                ctx.state,
                prompt,
                retry_count=self.retry_count,
                has_evaluation=bool(self.evaluation),
                complementary_info_num=ctx.state.complementary_info_num,
            )
            return Evaluate_UpdateDraft(result.output, retry_count=self.retry_count)
        
        except (RetryError, CircuitBreakerOpenError) as e:
            error_msg = _create_error_state_update("UpdateDraft", e)
            logger.error(error_msg, exc_info=e)
            ctx.state.decision_updated = error_msg
            return End(f"Agent execution failed: {str(e)}")
    

# Generation of Alternatives Node - Generates alternative options for decision-making.
@dataclass
class GenerationOfAlternatives(BaseNode[DecisionState]):
    """
    Generates alternative options for the decision.
    Supports re-evaluation with feedback loop.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Union[Evaluate_GenerationOfAlternatives, End]:
        _log_agent_start("GenerationOfAlternatives", bool(self.evaluation), self.retry_count)
        
        # Truncate inputs to prevent context overflow
        decision_draft_updated = truncate_text(ctx.state.decision_draft_updated, max_chars=2000)
        alternatives = truncate_text(ctx.state.alternatives, max_chars=2000)
        
        base_prompt = f"Here the decision requested by user: {decision_draft_updated}"
        
        if self.evaluation:
            evaluation_truncated = truncate_text(self.evaluation, max_chars=1000)
            prompt = (
                f"{base_prompt}\n"
                f"Here the current alternatives for this decision: {alternatives}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {evaluation_truncated}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        _log_prompt_debug("GenerationOfAlternatives", prompt)
        
        try:
            result = await run_generation_of_alternatives(
                ctx.state,
                prompt,
                retry_count=self.retry_count,
                has_evaluation=bool(self.evaluation),
            )
            return Evaluate_GenerationOfAlternatives(result.output, retry_count=self.retry_count)
        
        except (RetryError, CircuitBreakerOpenError) as e:
            error_msg = _create_error_state_update("GenerationOfAlternatives", e)
            logger.error(error_msg, exc_info=e)
            ctx.state.alternatives = error_msg
            return End(f"Agent execution failed: {str(e)}")
    
    
# Result Node - Final node to evaluate and present the decision outcome.
@dataclass
class Result(BaseNode[DecisionState]):
    """
    Evaluates and selects the best alternative for the decision.
    Produces final decision output with commentary.
    """
    
    evaluation: Optional[str] = None
    retry_count: int = 0
    
    async def run(self, ctx: GraphRunContext[DecisionState]) -> Union[Evaluate_Result, End]:
        _log_agent_start("Result", bool(self.evaluation), self.retry_count)
        
        # Truncate inputs to prevent context overflow
        decision_draft_updated = truncate_text(ctx.state.decision_draft_updated, max_chars=2000)
        alternatives = truncate_text(ctx.state.alternatives, max_chars=2000)
        result_text = truncate_text(ctx.state.result, max_chars=1000)
        result_comment = truncate_text(ctx.state.result_comment, max_chars=1000)
        best_alt_result = truncate_text(ctx.state.best_alternative_result, max_chars=1000)
        best_alt_comment = truncate_text(ctx.state.best_alternative_result_comment, max_chars=1000)
        
        base_prompt = (
            f"Here the decision requested by user: {decision_draft_updated}\n"
            f"Here the current alternatives for this decision: {alternatives}"
        )
        
        if self.evaluation:
            evaluation_truncated = truncate_text(self.evaluation, max_chars=1000)
            prompt = (
                f"{base_prompt}\n"
                f"Here the selected result for the decision: {result_text}\n"
                f"Here the comment on selected result for the decision: {result_comment}\n"
                f"Here the selected best alternative for the decision: {best_alt_result}\n"
                f"Here the comment on selected best alternative for the decision: {best_alt_comment}\n"
                f"You gave an answer but that was not correct.\n"
                f"Here the evaluation comments from your previous wrong answer: {evaluation_truncated}\n"
                f"Please fix it and give the correct answer."
            )
        else:
            prompt = base_prompt
        
        _log_prompt_debug("Result", prompt)
        
        try:
            result = await run_result_agent(
                ctx.state,
                prompt,
                retry_count=self.retry_count,
                has_evaluation=bool(self.evaluation),
            )
            return Evaluate_Result(result.output, retry_count=self.retry_count)
        
        except (RetryError, CircuitBreakerOpenError) as e:
            error_msg = _create_error_state_update("Result", e)
            logger.error(error_msg, exc_info=e)
            ctx.state.result = error_msg
            return End(f"Agent execution failed: {str(e)}")
    



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
        
        # Safely extract evaluation output with defensive checks
        is_correct, comment = _safe_eval_output(result)
        
        if is_correct or (self.retry_count >= settings.evaluator_max_retries):
            ctx.state.trigger = self.answer
            status = "correct" if is_correct else "max_retries_reached"
            
            # Flag quality concern when max retries reached
            if not is_correct:
                warning = f"IdentifyTrigger: Accepted after {self.retry_count} retries. Evaluator comment: {comment}"
                ctx.state.quality_warnings.append(warning)
                ctx.state.needs_human_review = True
                logger.warning(warning)
            
            _log_evaluation_result(
                node_name="Evaluate_IdentifyTrigger",
                status=status,
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return AnalyzeRootCause()
        else:
            _log_evaluation_result(
                node_name="Evaluate_IdentifyTrigger",
                status="retry",
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return IdentifyTrigger(evaluation=comment, retry_count=self.retry_count + 1)
        
        
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
        
        # Safely extract evaluation output with defensive checks
        is_correct, comment = _safe_eval_output(result)
        
        if is_correct or (self.retry_count >= settings.evaluator_max_retries):
            ctx.state.root_cause = self.answer
            status = "correct" if is_correct else "max_retries_reached"
            
            # Flag quality concern when max retries reached
            if not is_correct:
                warning = f"AnalyzeRootCause: Accepted after {self.retry_count} retries. Evaluator comment: {comment}"
                ctx.state.quality_warnings.append(warning)
                ctx.state.needs_human_review = True
                logger.warning(warning)
            
            _log_evaluation_result(
                node_name="Evaluate_AnalyzeRootCause",
                status=status,
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return ScopeDefinition()
        else:
            _log_evaluation_result(
                node_name="Evaluate_AnalyzeRootCause",
                status="retry",
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return AnalyzeRootCause(evaluation=comment, retry_count=self.retry_count + 1)


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
        
        # Safely extract evaluation output with defensive checks
        is_correct, comment = _safe_eval_output(result)
        
        if is_correct or (self.retry_count >= settings.evaluator_max_retries):
            ctx.state.scope_definition = self.answer
            status = "correct" if is_correct else "max_retries_reached"
            
            # Flag quality concern when max retries reached
            if not is_correct:
                warning = f"ScopeDefinition: Accepted after {self.retry_count} retries. Evaluator comment: {comment}"
                ctx.state.quality_warnings.append(warning)
                ctx.state.needs_human_review = True
                logger.warning(warning)
            
            _log_evaluation_result(
                node_name="Evaluate_ScopeDefinition",
                status=status,
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return Drafting()
        else:
            _log_evaluation_result(
                node_name="Evaluate_ScopeDefinition",
                status="retry",
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return ScopeDefinition(evaluation=comment, retry_count=self.retry_count + 1)


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
        
        # Safely extract evaluation output with defensive checks
        is_correct, comment = _safe_eval_output(result)
        
        if is_correct or (self.retry_count >= settings.evaluator_max_retries):
            ctx.state.decision_drafted = self.answer
            status = "correct" if is_correct else "max_retries_reached"
            
            # Flag quality concern when max retries reached
            if not is_correct:
                warning = f"Drafting: Accepted after {self.retry_count} retries. Evaluator comment: {comment}"
                ctx.state.quality_warnings.append(warning)
                ctx.state.needs_human_review = True
                logger.warning(warning)
            
            _log_evaluation_result(
                node_name="Evaluate_Drafting",
                status=status,
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return EstablishGoals()
        else:
            _log_evaluation_result(
                node_name="Evaluate_Drafting",
                status="retry",
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return Drafting(evaluation=comment, retry_count=self.retry_count + 1)


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
        
        # Safely extract evaluation output with defensive checks
        is_correct, comment = _safe_eval_output(result)
        
        if is_correct or (self.retry_count >= settings.evaluator_max_retries):
            ctx.state.goals = self.answer
            status = "correct" if is_correct else "max_retries_reached"
            
            # Flag quality concern when max retries reached
            if not is_correct:
                warning = f"EstablishGoals: Accepted after {self.retry_count} retries. Evaluator comment: {comment}"
                ctx.state.quality_warnings.append(warning)
                ctx.state.needs_human_review = True
                logger.warning(warning)
            
            _log_evaluation_result(
                node_name="Evaluate_EstablishGoals",
                status=status,
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return IdentifyInformationNeeded()
        else:
            _log_evaluation_result(
                node_name="Evaluate_EstablishGoals",
                status="retry",
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return EstablishGoals(evaluation=comment, retry_count=self.retry_count + 1)


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
        
        # Safely extract evaluation output with defensive checks
        is_correct, comment = _safe_eval_output(result)
        
        if is_correct or (ctx.state.complementary_info_num >= 3):
            status = "correct" if is_correct else "max_iterations_reached"
            _log_evaluation_result(
                node_name="Evaluate_IdentifyInformationNeeded",
                status=status,
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=ctx.state.complementary_info_num
            )
            return UpdateDraft()
        else:
            # Information needs identified, proceed to retrieve it
            # Store the info needed in state for retry persistence
            ctx.state.info_needed_current = self.answer
            _log_evaluation_result(
                node_name="Evaluate_IdentifyInformationNeeded",
                status="proceeding_to_retrieval",
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=ctx.state.complementary_info_num,
                info_field_name="Info Needed"
            )
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
                'info needed': ctx.state.info_needed_current,
                'retrieved information': self.answer
            })
        )
        
        # Safely extract evaluation output with defensive checks
        is_correct, comment = _safe_eval_output(result)
        
        if is_correct or (self.retry_count >= settings.evaluator_max_retries):
            # Information quality approved or max retries reached
            ctx.state.complementary_info += "\n" + self.answer
            ctx.state.complementary_info_num += 1
            status = "correct" if is_correct else "max_retries_reached"
            _log_evaluation_result(
                node_name="Evaluate_RetrieveInformationNeeded",
                status=status,
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count,
                info_field_name="Retrieved Info"
            )
            return IdentifyInformationNeeded(complementary_info=True)
        else:
            # Information needs improvement
            _log_evaluation_result(
                node_name="Evaluate_RetrieveInformationNeeded",
                status="revision_required",
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count,
                info_field_name="Retrieved Info"
            )
            # Use the persisted info_needed from state for retry consistency
            return RetrieveInformationNeeded(
                info_needed=ctx.state.info_needed_current,
                evaluation=comment
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
                'original decision draft': ctx.state.decision_drafted,
                'updated draft': self.answer,
                'complementary information': ctx.state.complementary_info
            })
        )
        
        # Safely extract evaluation output with defensive checks
        is_correct, comment = _safe_eval_output(result)
        
        if is_correct or (self.retry_count >= settings.evaluator_max_retries):
            ctx.state.decision_draft_updated = self.answer
            status = "correct" if is_correct else "max_retries_reached"
            
            # Flag quality concern when max retries reached
            if not is_correct:
                warning = f"UpdateDraft: Accepted after {self.retry_count} retries. Evaluator comment: {comment}"
                ctx.state.quality_warnings.append(warning)
                ctx.state.needs_human_review = True
                logger.warning(warning)
            
            _log_evaluation_result(
                node_name="Evaluate_UpdateDraft",
                status=status,
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return GenerationOfAlternatives()
        else:
            _log_evaluation_result(
                node_name="Evaluate_UpdateDraft",
                status="retry",
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return UpdateDraft(evaluation=comment, retry_count=self.retry_count + 1)
        
        
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
                'decision draft': ctx.state.decision_draft_updated,
                'generated alternatives': self.answer
            })
        )
        
        # Safely extract evaluation output with defensive checks
        is_correct, comment = _safe_eval_output(result)
        
        if is_correct or (self.retry_count >= settings.evaluator_max_retries):
            ctx.state.alternatives = self.answer
            status = "correct" if is_correct else "max_retries_reached"
            
            # Flag quality concern when max retries reached
            if not is_correct:
                warning = f"GenerationOfAlternatives: Accepted after {self.retry_count} retries. Evaluator comment: {comment}"
                ctx.state.quality_warnings.append(warning)
                ctx.state.needs_human_review = True
                logger.warning(warning)
            
            _log_evaluation_result(
                node_name="Evaluate_GenerationOfAlternatives",
                status=status,
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return Result()
        else:
            _log_evaluation_result(
                node_name="Evaluate_GenerationOfAlternatives",
                status="retry",
                answer=self.answer,
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return GenerationOfAlternatives(evaluation=comment, retry_count=self.retry_count + 1)
        

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
        
        result = await result_agent_evaluator.run(
            format_as_xml({
                'decision requested': ctx.state.decision_requested,
                'updated draft': ctx.state.decision_draft_updated,
                'alternatives': ctx.state.alternatives,
                'selected result': self.answer.result,
                'result comment': self.answer.result_comment,
                'best alternative': self.answer.best_alternative_result,
                'best alternative comment': self.answer.best_alternative_result_comment
            })
        )
        
        # Safely extract evaluation output with defensive checks
        is_correct, comment = _safe_eval_output(result)
        
        if is_correct or (self.retry_count >= settings.evaluator_max_retries):
            ctx.state.result = self.answer.result
            ctx.state.result_comment = self.answer.result_comment
            ctx.state.best_alternative_result = self.answer.best_alternative_result
            ctx.state.best_alternative_result_comment = self.answer.best_alternative_result_comment
            status = "correct" if is_correct else "max_retries_reached"
            _log_evaluation_result(
                node_name="Evaluate_Result",
                status=status,
                answer=str(self.answer.result),
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return End(True)
        else:
            _log_evaluation_result(
                node_name="Evaluate_Result",
                status="retry",
                answer=f"Result: {self.answer.result}, Alternative: {self.answer.best_alternative_result}",
                evaluation_comment=comment,
                retry_count=self.retry_count
            )
            return Result(evaluation=comment, retry_count=self.retry_count + 1)
