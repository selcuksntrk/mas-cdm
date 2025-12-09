import pytest

from backend.app.core.exceptions import (
    CircuitBreakerOpenError,
    LLMAPIConnectionError,
    LLMProviderError,
    RetryError,
)
from backend.app.core.resilience.retry import CircuitBreaker, RetryPolicy, execute_with_resilience


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failures():
    attempts = 0

    async def flaky_call():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise LLMAPIConnectionError("transient")
        return "ok"

    policy = RetryPolicy(
        max_attempts=3,
        backoff_factor=0,
        jitter=0,
        retryable_exceptions=(LLMAPIConnectionError,),
    )

    result = await execute_with_resilience(
        flaky_call,
        policy=policy,
        circuit_breaker=CircuitBreaker(failure_threshold=5, recovery_time=1),
    )

    assert result == "ok"
    assert attempts == 3


@pytest.mark.asyncio
async def test_retry_exhaustion_raises_retry_error():
    async def always_fail():
        raise LLMProviderError("persistent failure")

    policy = RetryPolicy(
        max_attempts=2,
        backoff_factor=0,
        jitter=0,
        retryable_exceptions=(LLMProviderError,),
    )

    with pytest.raises(RetryError):
        await execute_with_resilience(
            always_fail,
            policy=policy,
            circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_time=10),
        )


@pytest.mark.asyncio
async def test_circuit_breaker_blocks_when_open():
    breaker = CircuitBreaker(failure_threshold=2, recovery_time=60)

    async def failing_call():
        raise LLMProviderError("boom")

    # First call increments failure_count to 1
    with pytest.raises(RetryError):
        await execute_with_resilience(
            failing_call,
            policy=RetryPolicy(max_attempts=1, backoff_factor=0, jitter=0, retryable_exceptions=(LLMProviderError,)),
            circuit_breaker=breaker,
        )

    # Second call trips the breaker to open
    with pytest.raises(RetryError):
        await execute_with_resilience(
            failing_call,
            policy=RetryPolicy(max_attempts=1, backoff_factor=0, jitter=0, retryable_exceptions=(LLMProviderError,)),
            circuit_breaker=breaker,
        )

    # Third call is blocked immediately by the breaker
    with pytest.raises(CircuitBreakerOpenError):
        await execute_with_resilience(
            failing_call,
            policy=RetryPolicy(max_attempts=1, backoff_factor=0, jitter=0, retryable_exceptions=(LLMProviderError,)),
            circuit_breaker=breaker,
        )


@pytest.mark.asyncio
async def test_fallback_invoked_after_primary_failures():
    primary_calls = 0
    fallback_calls = 0

    async def primary():
        nonlocal primary_calls
        primary_calls += 1
        raise LLMAPIConnectionError("primary down")

    async def fallback():
        nonlocal fallback_calls
        fallback_calls += 1
        return "fallback-success"

    policy = RetryPolicy(
        max_attempts=1,
        backoff_factor=0,
        jitter=0,
        retryable_exceptions=(LLMAPIConnectionError,),
    )

    result = await execute_with_resilience(
        primary,
        fallback=fallback,
        policy=policy,
        circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_time=10),
    )

    assert result == "fallback-success"
    assert primary_calls == 1
    assert fallback_calls == 1
