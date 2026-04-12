"""
Simple async-safe circuit breaker for the Anthropic API client.

States:
  CLOSED   — normal operation; failures are counted
  OPEN     — Anthropic is degraded; requests are rejected immediately
  HALF_OPEN — after reset timeout, one probe request is allowed through;
               if it succeeds the breaker closes, if it fails it re-opens

Why this matters:
  The Anthropic API can be slow or unavailable. Without a circuit breaker,
  every Scout request blocks for the full httpx timeout (~60 s) before failing,
  which starves the Uvicorn thread pool and causes cascading timeouts across
  the entire API. The breaker fails fast so users get an immediate error
  instead of a timeout, and the system stays responsive.

Usage:
    breaker = AnthropicCircuitBreaker()

    async with breaker:          # raises CircuitOpenError if OPEN
        response = await client.messages.create(...)
        breaker.record_success()

    # or check explicitly:
    if not breaker.allow_request():
        raise CircuitOpenError(...)
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from enum import Enum, auto

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED    = auto()
    OPEN      = auto()
    HALF_OPEN = auto()


class CircuitOpenError(RuntimeError):
    """Raised when a request is rejected because the circuit breaker is open."""


class AnthropicCircuitBreaker:
    """
    Thread-safe, async-compatible circuit breaker.

    Parameters
    ----------
    failure_threshold : int
        Number of failures within *window_seconds* that trips the breaker OPEN.
    window_seconds : float
        Rolling time window for counting failures.
    reset_timeout : float
        Seconds to wait in OPEN state before allowing one probe (HALF_OPEN).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        window_seconds: float = 60.0,
        reset_timeout: float = 30.0,
    ) -> None:
        self._threshold   = failure_threshold
        self._window      = window_seconds
        self._reset_timeout = reset_timeout

        self._state = CircuitState.CLOSED
        self._failures: deque[float] = deque()   # timestamps of recent failures
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def allow_request(self) -> bool:
        """Return True if a request should be allowed through."""
        async with self._lock:
            return self._check_state()

    async def record_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("circuit_breaker.closed — probe succeeded")
                self._state = CircuitState.CLOSED
                self._failures.clear()
                self._opened_at = None

    async def record_failure(self) -> None:
        from app.observability.metrics import SCOUT_CIRCUIT_BREAKER_OPEN
        async with self._lock:
            now = time.monotonic()
            self._failures.append(now)
            self._evict_old_failures(now)

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — re-open immediately
                self._state = CircuitState.OPEN
                self._opened_at = now
                logger.warning("circuit_breaker.reopened — probe failed")
                SCOUT_CIRCUIT_BREAKER_OPEN.inc()
            elif len(self._failures) >= self._threshold:
                self._state = CircuitState.OPEN
                self._opened_at = now
                logger.error(
                    "circuit_breaker.opened failures=%d window=%.0fs",
                    len(self._failures), self._window,
                )
                SCOUT_CIRCUIT_BREAKER_OPEN.inc()

    # ── Context manager (convenience) ─────────────────────────────────────────

    async def __aenter__(self):
        from app.observability.metrics import SCOUT_CIRCUIT_BREAKER_REJECTED
        if not await self.allow_request():
            SCOUT_CIRCUIT_BREAKER_REJECTED.inc()
            raise CircuitOpenError(
                "The Scout analyst is temporarily unavailable — Anthropic API degraded. "
                "Please try again in a moment."
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self.record_success()
        else:
            await self.record_failure()
        return False   # don't suppress exceptions

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_state(self) -> bool:
        """Must be called with self._lock held."""
        now = time.monotonic()

        if self._state == CircuitState.CLOSED:
            self._evict_old_failures(now)
            return True

        if self._state == CircuitState.OPEN:
            assert self._opened_at is not None
            if now - self._opened_at >= self._reset_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("circuit_breaker.half_open — allowing probe")
                return True
            return False

        # HALF_OPEN — already letting one probe through; block others
        return False

    def _evict_old_failures(self, now: float) -> None:
        cutoff = now - self._window
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()


# Module-level singleton — shared across all Scout calls in the process.
anthropic_breaker = AnthropicCircuitBreaker(
    failure_threshold=5,
    window_seconds=60.0,
    reset_timeout=30.0,
)
