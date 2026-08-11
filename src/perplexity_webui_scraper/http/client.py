"""HTTP client wrapping curl-cffi with retry, rate limiting, and error handling.

``HTTPClient`` is the single transport layer for all Perplexity API calls.
It manages a persistent curl-cffi ``Session`` with browser impersonation,
applies rate limiting and exponential-backoff retry, and translates HTTP
error codes into typed exceptions.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite
from time import monotonic
from typing import TYPE_CHECKING, Any

from curl_cffi.requests import Response as CurlResponse
from curl_cffi.requests import Session
from curl_cffi.requests import exceptions as curl_exceptions

from perplexity_webui_scraper._internal.constants import (
    API_BASE_URL,
    DEFAULT_HEADERS,
    DEFAULT_TIMEOUT,
    ENDPOINT_ASK,
    ENDPOINT_SEARCH_INIT,
    SESSION_COOKIE_NAME,
)
from perplexity_webui_scraper._internal.exceptions import (
    AuthenticationError,
    HTTPError,
    PerplexityError,
    RateLimitError,
    TransientHTTPError,
)
from perplexity_webui_scraper._internal.logging import (
    get_logger,
    log_request,
    log_response,
    log_retry,
)
from perplexity_webui_scraper.http.fingerprint import get_random_browser_profile
from perplexity_webui_scraper.http.resilience import RateLimiter, RetryConfig, retry_with_backoff


if TYPE_CHECKING:
    from collections.abc import Generator

    from curl_cffi.requests import BrowserTypeLiteral


logger = get_logger(__name__)


def _parse_retry_after(value: object) -> float | None:
    """Parse delta-seconds or HTTP-date Retry-After header safely."""
    if not isinstance(value, str):
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = (retry_at - datetime.now(UTC)).total_seconds()
    return seconds if isfinite(seconds) and seconds >= 0 else None


class HTTPClient:
    """HTTP client with retry, rate limiting, fingerprint rotation, and error handling.

    Attributes:
        _session_token: The Perplexity session token used for authentication.
        _timeout: Request timeout in seconds.
        _impersonate: Current browser fingerprint profile.
        _rotate_fingerprint: Whether to rotate the profile on retry.
        _max_init_query_length: Max characters for the search-init query.
        _retry_config: Retry behaviour configuration.
        _rate_limiter: Optional rate limiter instance.
        _session: Active curl-cffi ``Session``.
    """

    __slots__ = (
        "_impersonate",
        "_max_init_query_length",
        "_rate_limiter",
        "_retry_config",
        "_rotate_fingerprint",
        "_session",
        "_session_token",
        "_timeout",
    )

    def __init__(
        self,
        session_token: str,
        timeout: int = DEFAULT_TIMEOUT,
        impersonate: BrowserTypeLiteral = "chrome",
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
        retry_jitter: float = 0.5,
        max_rate_limit_delay: float = 300.0,
        circuit_failure_threshold: int = 3,
        circuit_cooldown: float = 30.0,
        requests_per_second: float = 0.5,
        rate_limiter: RateLimiter | None = None,
        rotate_fingerprint: bool = True,
        max_init_query_length: int = 2000,
    ) -> None:
        """Initialise the HTTP client.

        Args:
            session_token: The ``__Secure-next-auth.session-token`` cookie value.
            timeout: Request timeout in seconds.
            impersonate: Initial browser fingerprint profile.
            max_retries: Maximum retry attempts on transient errors.
            retry_base_delay: Initial backoff delay in seconds.
            retry_max_delay: Maximum backoff delay cap in seconds.
            retry_jitter: Jitter factor (0-1).
            max_rate_limit_delay: Maximum Retry-After delay accepted in seconds.
            circuit_failure_threshold: Consecutive transient failures before opening circuit.
            circuit_cooldown: Minimum circuit-open cooldown in seconds.
            requests_per_second: Rate limit; ``0`` disables it.
            rate_limiter: Optional shared pacing and cooldown state.
            rotate_fingerprint: Rotate fingerprint on each retry.
            max_init_query_length: Truncate init query to this length;
                ``0`` disables truncation.
        """
        self._session_token = session_token
        self._timeout = timeout
        self._impersonate: BrowserTypeLiteral = impersonate
        self._rotate_fingerprint = rotate_fingerprint
        self._max_init_query_length = max_init_query_length

        self._retry_config = RetryConfig(
            max_retries=max_retries,
            base_delay=retry_base_delay,
            max_delay=retry_max_delay,
            jitter=retry_jitter,
            max_rate_limit_delay=max_rate_limit_delay,
        )

        self._rate_limiter = rate_limiter or (
            RateLimiter(
                requests_per_second=requests_per_second,
                circuit_failure_threshold=circuit_failure_threshold,
                circuit_cooldown=circuit_cooldown,
            )
            if requests_per_second > 0
            else None
        )

        self._session = self._create_session(impersonate)
        logger.debug("HTTPClient initialized | impersonate={}", impersonate)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _create_session(self, impersonate: BrowserTypeLiteral) -> Session:
        """Create a new curl-cffi session with auth cookie and default headers.

        Args:
            impersonate: Browser profile to impersonate.

        Returns:
            Configured :class:`curl_cffi.requests.Session`.
        """
        return Session(
            headers=dict(DEFAULT_HEADERS),
            cookies={SESSION_COOKIE_NAME: self._session_token},
            timeout=self._timeout,
            impersonate=impersonate,  # type: ignore[arg-type]
        )

    def _rotate_session(self) -> None:
        """Replace the current session with a new random browser fingerprint."""
        if not self._rotate_fingerprint:
            return

        new_profile = get_random_browser_profile()
        logger.debug("Rotating fingerprint | old={} new={}", self._impersonate, new_profile)

        with suppress(Exception):
            self._session.close()

        self._impersonate = new_profile
        self._session = self._create_session(new_profile)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self, *, pace: bool = True) -> None:
        """Apply shared provider cooldown and optional normal request pacing."""
        if self._rate_limiter:
            self._rate_limiter.acquire(pace=pace)

    def record_rate_limit(self, error: RateLimitError) -> None:
        """Record bounded provider throttling discovered outside status handling."""
        if self._rate_limiter:
            retry_after = error.retry_after
            if retry_after is not None:
                retry_after = min(retry_after, self._retry_config.max_rate_limit_delay)
            self._rate_limiter.record_rate_limit(retry_after)

    def _record_transient_failure(self, error: Exception) -> None:
        """Record one retryable upstream failure in shared circuit state."""
        if self._rate_limiter is None:
            return

        retry_after = error.retry_after if isinstance(error, RateLimitError) else None
        if retry_after is not None:
            retry_after = min(retry_after, self._retry_config.max_rate_limit_delay)
        self._rate_limiter.record_transient_failure(retry_after)

    def _on_retry(self, attempt: int, exception: Exception, wait: float) -> None:
        """Callback invoked before each retry attempt.

        Args:
            attempt: Current attempt number (1-based).
            exception: The exception that triggered the retry.
            wait: Seconds to wait before the next attempt.
        """
        log_retry(attempt, self._retry_config.max_retries, exception, wait)

        if self._rotate_fingerprint and not isinstance(exception, RateLimitError):
            self._rotate_session()

    def _handle_error(self, error: Exception, context: str = "") -> None:
        """Translate a raw exception into a typed Perplexity exception.

        Args:
            error: The original exception from curl-cffi.
            context: Optional prefix describing the request context.

        Raises:
            AuthenticationError: On HTTP 403.
            RateLimitError: On HTTP 429.
            HTTPError: On any other HTTP error with a status code.
            PerplexityError: On network-level errors without a status code.
        """
        response = getattr(error, "response", None)
        status_code: int | None = None
        url: str | None = None
        response_body: str | None = None

        retry_after: float | None = None
        if response is not None:
            status_code = getattr(response, "status_code", None)
            url = getattr(response, "url", None)
            headers = getattr(response, "headers", {})
            retry_after = _parse_retry_after(getattr(headers, "get", lambda _key: None)("Retry-After"))

            with suppress(Exception):
                response_body = response.text if hasattr(response, "text") else None

        match status_code:
            case 401 | 403:
                raise AuthenticationError from error
            case 429:
                rate_error = RateLimitError(
                    url=str(url) if url else None,
                    response_body=response_body,
                    retry_after=retry_after,
                )
                raise rate_error from error
            case _ if status_code is not None and status_code >= 500:
                raise TransientHTTPError(
                    f"{context}HTTP {status_code}: {error!s}",
                    status_code=status_code,
                    url=str(url) if url else None,
                    response_body=response_body,
                ) from error
            case _ if status_code is not None:
                raise HTTPError(
                    f"{context}HTTP {status_code}: {error!s}",
                    status_code=status_code,
                    url=str(url) if url else None,
                    response_body=response_body,
                ) from error
            case _:
                raise PerplexityError(f"{context}{error!s}") from error

    def _raise_for_status(self, response: CurlResponse, context: str) -> None:
        """Raise typed project exceptions for non-success HTTP responses."""
        try:
            response.raise_for_status()
        except Exception as error:
            self._handle_error(error, context)

    # ------------------------------------------------------------------
    # Public request methods
    # ------------------------------------------------------------------

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        rate_limited: bool = True,
    ) -> CurlResponse:
        """Make a GET request with retry and rate limiting.

        Args:
            endpoint: Relative path (e.g. ``"/search/new"``) or full URL.
            params: Optional query parameters.
            rate_limited: Whether to apply the configured request rate limiter.

        Returns:
            The curl-cffi response object.

        Raises:
            AuthenticationError: On HTTP 403.
            RateLimitError: On HTTP 429.
            HTTPError: On other HTTP errors.
            PerplexityError: On network errors.
        """
        url = f"{API_BASE_URL}{endpoint}" if endpoint.startswith("/") else endpoint
        log_request("GET", url, params=params)

        def _do_get() -> CurlResponse:
            self._throttle(pace=rate_limited)
            t0 = monotonic()
            response = self._session.get(url, params=params)
            log_response("GET", url, response.status_code, elapsed_ms=(monotonic() - t0) * 1000)
            self._raise_for_status(response, f"GET {endpoint}: ")
            if self._rate_limiter:
                self._rate_limiter.record_success()

            return response

        try:
            return retry_with_backoff(
                _do_get,
                self._retry_config,
                on_retry=self._on_retry,
                on_failure=self._record_transient_failure,
                retryable=(
                    RateLimitError,
                    TransientHTTPError,
                    curl_exceptions.ConnectionError,
                    curl_exceptions.Timeout,
                ),
            )
        except (RateLimitError, AuthenticationError, HTTPError, PerplexityError):
            raise
        except Exception as error:
            self._handle_error(error, f"GET {endpoint}: ")
            raise  # unreachable but satisfies type checker

    def post(
        self,
        endpoint: str,
        json: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> CurlResponse:
        """Make a POST request with retry and rate limiting.

        Args:
            endpoint: Relative path or full URL.
            json: Optional JSON body to serialize and send.
            stream: If ``True``, keep the response connection open for streaming.

        Returns:
            The curl-cffi response object.

        Raises:
            AuthenticationError: On HTTP 403.
            RateLimitError: On HTTP 429.
            HTTPError: On other HTTP errors.
            PerplexityError: On network errors.
        """
        url = f"{API_BASE_URL}{endpoint}" if endpoint.startswith("/") else endpoint
        log_request("POST", url, body_size=len(str(json)) if json else 0)

        def _do_post() -> CurlResponse:
            self._throttle()
            t0 = monotonic()
            response = self._session.post(url, json=json, stream=stream)
            log_response("POST", url, response.status_code, elapsed_ms=(monotonic() - t0) * 1000)
            self._raise_for_status(response, f"POST {endpoint}: ")
            if self._rate_limiter:
                self._rate_limiter.record_success()

            return response

        try:
            return retry_with_backoff(
                _do_post,
                self._retry_config,
                on_retry=self._on_retry,
                on_failure=self._record_transient_failure,
                retryable=(
                    RateLimitError,
                    TransientHTTPError,
                    curl_exceptions.ConnectionError,
                    curl_exceptions.Timeout,
                ),
            )
        except (RateLimitError, AuthenticationError, HTTPError, PerplexityError):
            raise
        except Exception as error:
            self._handle_error(error, f"POST {endpoint}: ")
            raise  # unreachable but satisfies type checker

    def _stream_lines(self, endpoint: str, json: dict[str, Any]) -> Generator[bytes, None, None]:
        """Make a streaming POST and yield raw SSE lines as bytes.

        Args:
            endpoint: Relative path or full URL.
            json: JSON payload.

        Yields:
            Raw bytes lines from the SSE response.
        """
        response = self.post(endpoint, json=json, stream=True)

        try:
            yield from response.iter_lines()
        except (curl_exceptions.ConnectionError, curl_exceptions.Timeout) as error:
            self._record_transient_failure(error)
            raise TransientHTTPError(f"Stream {endpoint}: {error!s}") from error
        finally:
            response.close()

    def init_search(self, query: str) -> None:
        """Initialize a search session (required before each prompt).

        The query is sent as a GET parameter.  Very long queries can exceed
        server URI limits (HTTP 414).  When ``max_init_query_length > 0``,
        the query is truncated to stay within safe limits.

        Args:
            query: The search query string to initialize.
        """
        if self._max_init_query_length and len(query) > self._max_init_query_length:
            query = query[: self._max_init_query_length]

        self.get(ENDPOINT_SEARCH_INIT, params={"q": query})

    def stream_ask(self, payload: dict[str, Any]) -> Generator[bytes, None, None]:
        """Stream a prompt request to the Perplexity SSE ask endpoint.

        Args:
            payload: The fully-constructed request payload (see
                :func:`~perplexity_webui_scraper.core.payload.build_payload`).

        Yields:
            Raw bytes lines from the SSE stream.
        """
        yield from self._stream_lines(ENDPOINT_ASK, json=payload)

    def close(self) -> None:
        """Close the underlying curl-cffi session and release resources."""
        self._session.close()

    def __enter__(self) -> HTTPClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
