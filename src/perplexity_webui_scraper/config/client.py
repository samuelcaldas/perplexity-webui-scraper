"""HTTP client and resilience configuration model."""

from __future__ import annotations

from os import PathLike  # noqa: TC003

from curl_cffi.requests import BrowserTypeLiteral  # noqa: TC002
from pydantic import BaseModel, ConfigDict

from perplexity_webui_scraper._internal.types import LogLevel  # noqa: TC001


class ClientConfig(BaseModel):
    """Immutable HTTP client and resilience settings.

    Attributes:
        timeout: Request timeout in seconds. Defaults to 3600 (1 hour) for long queries.
        impersonate: Browser fingerprint profile (e.g., ``"chrome"``, ``"firefox"``).
        max_retries: Max retry attempts on transient errors.
        retry_base_delay: Initial backoff delay in seconds.
        retry_max_delay: Maximum backoff delay cap in seconds.
        retry_jitter: Jitter factor (0-1) to randomize retry delays.
        max_rate_limit_delay: Maximum accepted provider Retry-After delay.
        circuit_failure_threshold: Consecutive transient failures before fail-fast mode.
        circuit_cooldown: Minimum circuit-open duration in seconds.
        rotate_fingerprint: Rotates browser fingerprint on each retry if ``True``.
        requests_per_second: Max request rate. Set to ``0`` to disable.
        account_profile_ttl: Seconds before account entitlement data is refreshed.
        max_init_query_length: Truncates init query length to avoid HTTP 414.
        logging_level: Log verbosity. Defaults to ``"disabled"``.
        log_file: Path to write logs. ``None`` writes to stderr.
    """

    model_config = ConfigDict(frozen=True)

    timeout: int = 3600
    impersonate: BrowserTypeLiteral = "chrome"
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_jitter: float = 0.5
    max_rate_limit_delay: float = 300.0
    circuit_failure_threshold: int = 3
    circuit_cooldown: float = 30.0
    rotate_fingerprint: bool = True
    requests_per_second: float = 0.5
    account_profile_ttl: float = 60.0
    max_init_query_length: int = 2000
    logging_level: LogLevel = "disabled"
    log_file: str | PathLike[str] | None = None
