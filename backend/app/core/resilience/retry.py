"""
Resilience utilities: retries with backoff, circuit breaker, and fallback execution.

Designed to shield agent execution from transient failures (rate limits,
network hiccups) while preventing cascading failures via a circuit breaker.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Sequence, Tuple, Type, TypeVar

from backend.app.core.exceptions import CircuitBreakerOpenError, RetryError


logger = logging.getLogger(__name__)

T = TypeVar("T")


# Retry policy configuration
@dataclass
class RetryPolicy:
    max_attempts: int = 3
    backoff_factor: float = 0.5
    max_backoff: float = 8.0
    jitter: float = 0.2
    retryable_exceptions: Tuple[Type[BaseException], ...] = (Exception,)


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_time: float = 30.0
    half_open_success_threshold: int = 1
    _state: str = field(default="closed", init=False)
    _failure_count: int = field(default=0, init=False)
    _opened_at: Optional[float] = field(default=None, init=False)
    _half_open_successes: int = field(default=0, init=False)

    def ensure_can_execute(self) -> None:
        now = time.monotonic()
        if self._state == "open":
            if self._opened_at is None:
                self._opened_at = now
            elapsed = now - self._opened_at
            if elapsed >= self.recovery_time:
                # Move to half-open to allow a probe call
                self._state = "half_open"
                self._half_open_successes = 0
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker open; retry after {self.recovery_time - elapsed:.1f}s"
                )

    def record_success(self) -> None:
        if self._state in ("half_open", "open"):
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_success_threshold:
                self._close()
        else:
            self._reset_failure_count()

    def record_failure(self) -> None:
        if self._state == "half_open":
            # Immediately open on failure during half-open probe
            self._open()
            return

        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._open()

    def _open(self) -> None:
        self._state = "open"
        self._opened_at = time.monotonic()
        logger.warning("Circuit breaker opened after %s consecutive failures", self._failure_count)

    def _close(self) -> None:
        self._state = "closed"
        self._opened_at = None
        self._half_open_successes = 0
        self._reset_failure_count()
        logger.info("Circuit breaker closed; service considered healthy again")

    def _reset_failure_count(self) -> None:
        self._failure_count = 0


async def _retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    circuit_breaker: Optional[CircuitBreaker],
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> T:
    attempt = 1
    last_error: Optional[BaseException] = None

    while attempt <= policy.max_attempts:
        if circuit_breaker:
            circuit_breaker.ensure_can_execute()

        try:
            result = await func()
            if circuit_breaker:
                circuit_breaker.record_success()
            return result
        except policy.retryable_exceptions as exc:  # type: ignore[misc]
            last_error = exc
            if circuit_breaker:
                circuit_breaker.record_failure()

            if attempt >= policy.max_attempts:
                break

            delay = _compute_backoff(policy, attempt)
            if on_retry:
                on_retry(attempt, exc)
            logger.warning(
                "Retrying after failure (attempt %s/%s, delay %.2fs): %s",
                attempt,
                policy.max_attempts,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
            attempt += 1
            continue
        except Exception:
            # Non-retryable error bubbles up immediately
            if circuit_breaker:
                circuit_breaker.record_failure()
            raise

    raise RetryError(f"Operation failed after {policy.max_attempts} attempts") from last_error


def _compute_backoff(policy: RetryPolicy, attempt: int) -> float:
    base = policy.backoff_factor * (2 ** (attempt - 1))
    jitter = random.uniform(0, policy.jitter)
    return min(base + jitter, policy.max_backoff)


async def execute_with_resilience(
    primary: Callable[[], Awaitable[T]],
    *,
    fallback: Optional[Callable[[], Awaitable[T]]] = None,
    policy: RetryPolicy | None = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> T:
    """
    Execute a coroutine with retry/backoff and an optional fallback callable.

    Args:
        primary: Coroutine factory for the primary call.
        fallback: Optional coroutine factory for a fallback (e.g., secondary model).
        policy: Retry policy (uses sensible defaults if omitted).
        circuit_breaker: Optional circuit breaker to guard repeated failures.
        on_retry: Optional hook invoked before each retry with (attempt, error).

    Returns:
        Result of the successful call.

    Raises:
        CircuitBreakerOpenError: If the breaker is open and cannot execute.
        RetryError: If all attempts (including fallback) are exhausted.
        Exception: Any non-retryable exception from the call surfaces.
    """

    retry_policy = policy or RetryPolicy()

    try:
        return await _retry_async(primary, policy=retry_policy, circuit_breaker=circuit_breaker, on_retry=on_retry)
    except (RetryError, CircuitBreakerOpenError):
        # Bubble up immediately if breaker is open or retries exhausted with no fallback
        if not fallback:
            raise
        logger.info("Primary execution failed; attempting fallback path")
    except Exception:
        # For non-retryable errors try fallback if present
        if not fallback:
            raise
        logger.info("Primary execution raised non-retryable error; attempting fallback path", exc_info=True)

    # Attempt fallback path with fresh retry cycle; breaker is reused to avoid overload
    if not fallback:
        raise RetryError("Fallback not provided for failed primary execution")

    return await _retry_async(fallback, policy=retry_policy, circuit_breaker=circuit_breaker, on_retry=on_retry)
