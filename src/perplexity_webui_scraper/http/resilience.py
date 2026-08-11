"""Retry and shared rate-limit utilities for HTTP transport."""

from __future__ import annotations

from random import uniform
from threading import Lock
from time import monotonic, sleep
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ConfigDict

from perplexity_webui_scraper._internal.exceptions import CircuitOpenError, RateLimitError


if TYPE_CHECKING:
    from collections.abc import Callable


T = TypeVar("T")


class RetryConfig(BaseModel):
    """Immutable configuration for exponential-backoff retry behaviour."""

    model_config = ConfigDict(frozen=True)

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.5
    max_rate_limit_delay: float = 300.0


class RateLimiter:
    """Thread-safe request pacing, cooldown, and transient-failure circuit state."""

    __slots__ = (
        "_circuit_open_until",
        "_consecutive_failures",
        "_cooldown_until",
        "_half_open",
        "_last_request",
        "_lock",
        "circuit_cooldown",
        "circuit_failure_threshold",
        "requests_per_second",
    )

    def __init__(
        self,
        requests_per_second: float = 0.5,
        circuit_failure_threshold: int = 3,
        circuit_cooldown: float = 30.0,
    ) -> None:
        """Create shared provider pacing and bounded circuit state."""
        if circuit_failure_threshold < 1:
            raise ValueError("circuit_failure_threshold must be positive")
        if circuit_cooldown < 0:
            raise ValueError("circuit_cooldown must not be negative")

        self.requests_per_second = requests_per_second
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_cooldown = circuit_cooldown
        self._last_request = 0.0
        self._cooldown_until = 0.0
        self._circuit_open_until = 0.0
        self._consecutive_failures = 0
        self._half_open = False
        self._lock = Lock()

    def acquire(self, *, pace: bool = True) -> None:
        """Reserve request slot or fail fast while transient circuit remains open."""
        wait_time = self._reserve_request(pace)
        if wait_time > 0:
            sleep(wait_time)

    def _reserve_request(self, pace: bool) -> float:
        """Atomically validate circuit state and reserve normal pacing interval."""
        with self._lock:
            now = monotonic()
            self._raise_if_open(now)
            self._open_half_probe(now)
            wait_until = self._cooldown_until
            if pace and self.requests_per_second > 0:
                wait_until = max(wait_until, self._last_request + 1.0 / self.requests_per_second)
            if pace:
                self._last_request = max(now, wait_until)
            return max(0.0, wait_until - now)

    def _raise_if_open(self, now: float) -> None:
        """Reject all but one half-open probe before circuit recovery succeeds."""
        if self._circuit_open_until > now:
            raise CircuitOpenError(retry_after=self._circuit_open_until - now)
        if self._circuit_open_until and self._half_open:
            raise CircuitOpenError(retry_after=0.0)

    def _open_half_probe(self, now: float) -> None:
        """Reserve exactly one probe when elapsed circuit cooldown permits it."""
        if self._circuit_open_until and self._circuit_open_until <= now:
            self._half_open = True

    def record_rate_limit(self, retry_after: float | None) -> None:
        """Record trusted provider throttling as cooldown and transient failure."""
        self.record_transient_failure(retry_after)

    def record_transient_failure(self, retry_after: float | None = None) -> None:
        """Open bounded circuit after consecutive provider or transport failures."""
        with self._lock:
            now = monotonic()
            if retry_after is not None:
                self._cooldown_until = max(self._cooldown_until, now + retry_after)
            self._consecutive_failures += 1
            self._half_open = False
            if self._consecutive_failures < self.circuit_failure_threshold:
                return
            self._circuit_open_until = max(
                self._circuit_open_until,
                self._cooldown_until,
                now + self.circuit_cooldown,
            )

    def record_success(self) -> None:
        """Reset transient circuit after a successful upstream response."""
        with self._lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
            self._half_open = False
            if self._cooldown_until <= monotonic():
                self._cooldown_until = 0.0


def retry_with_backoff(
    fn: Callable[[], T],
    config: RetryConfig,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    on_failure: Callable[[Exception], None] | None = None,
    retryable: tuple[type[Exception], ...] = (),
) -> T:
    """Execute a callable with bounded server-aware exponential retry."""
    last_exc: Exception | None = None

    for attempt in range(1, config.max_retries + 2):
        try:
            return fn()
        except Exception as exc:
            if isinstance(exc, CircuitOpenError):
                raise
            if retryable and not isinstance(exc, retryable):
                raise
            if on_failure is not None:
                on_failure(exc)
            last_exc = exc

        if attempt > config.max_retries:
            break

        wait = _retry_delay(last_exc, attempt, config)
        if on_retry is not None:
            on_retry(attempt, last_exc, wait)
        sleep(wait)

    assert last_exc is not None
    raise last_exc


def _retry_delay(exception: Exception, attempt: int, config: RetryConfig) -> float:
    """Return capped provider hint or jittered exponential retry delay."""
    if isinstance(exception, RateLimitError) and exception.retry_after is not None:
        return min(exception.retry_after, config.max_rate_limit_delay)

    delay = min(config.base_delay * (2 ** (attempt - 1)), config.max_delay)
    jitter_amount = delay * config.jitter
    return max(0.0, delay + uniform(-jitter_amount, jitter_amount))
