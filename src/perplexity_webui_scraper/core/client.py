"""Perplexity top-level client class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from perplexity_webui_scraper._internal.constants import ENDPOINT_AUTH_SESSION, ENDPOINT_USER_SETTINGS
from perplexity_webui_scraper._internal.logging import configure_logging, get_logger
from perplexity_webui_scraper.config.client import ClientConfig
from perplexity_webui_scraper.config.conversation import ConversationConfig
from perplexity_webui_scraper.core.account import (
    AccountProfile,
    AccountProfileProvider,
    AccountSession,
    AccountSettings,
)
from perplexity_webui_scraper.core.conversation import Conversation
from perplexity_webui_scraper.http.client import HTTPClient


if TYPE_CHECKING:
    from perplexity_webui_scraper.http.resilience import RateLimiter


logger = get_logger(__name__)


class Perplexity:
    """Web scraper client for Perplexity AI conversations.

    The primary entry point. Create a single instance per session token and reuse it
    to share the underlying HTTP session and rate limiter.

    Example:
        ```python
        with Perplexity(session_token="...") as client:
            conversation = client.create_conversation()
            conversation.ask("Hello, world!")
            print(conversation.answer)
        ```

    Args:
        session_token: The ``__Secure-next-auth.session-token`` cookie value.
            Obtained via the ``get-session-token`` CLI tool.
        config: Optional client settings (timeouts, retries, logging).

    Raises:
        ValueError: If ``session_token`` is empty.
    """

    __slots__ = ("_account_profile_provider", "_http")

    def __init__(
        self,
        session_token: str,
        config: ClientConfig | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        if not session_token or not session_token.strip():
            raise ValueError("session_token cannot be empty")

        cfg = config or ClientConfig()
        configure_logging(level=cfg.logging_level, log_file=cfg.log_file)

        self._http = HTTPClient(
            session_token,
            timeout=cfg.timeout,
            impersonate=cfg.impersonate,
            max_retries=cfg.max_retries,
            retry_base_delay=cfg.retry_base_delay,
            retry_max_delay=cfg.retry_max_delay,
            retry_jitter=cfg.retry_jitter,
            max_rate_limit_delay=cfg.max_rate_limit_delay,
            circuit_failure_threshold=cfg.circuit_failure_threshold,
            circuit_cooldown=cfg.circuit_cooldown,
            requests_per_second=cfg.requests_per_second,
            rate_limiter=rate_limiter,
            rotate_fingerprint=cfg.rotate_fingerprint,
            max_init_query_length=cfg.max_init_query_length,
        )
        self._account_profile_provider = AccountProfileProvider(
            self.get_account_session,
            self.get_account_settings,
            ttl=cfg.account_profile_ttl,
        )

        logger.info("Perplexity client initialized")

    def create_conversation(
        self,
        config: ConversationConfig | None = None,
    ) -> Conversation:
        """Create and return a new :class:`~perplexity_webui_scraper.Conversation`.

        Args:
            config: Optional per-conversation settings.  Defaults to
                :class:`~perplexity_webui_scraper.config.ConversationConfig`
                defaults.

        Returns:
            A new :class:`~perplexity_webui_scraper.Conversation` instance
            ready to receive queries.
        """
        return Conversation(self._http, config or ConversationConfig(), self.get_account_profile)

    def get_account_session(self) -> AccountSession:
        """Return typed account/session information for the current token.

        This reads Perplexity's ``/api/auth/session`` endpoint and normalizes
        the account tier into ``free``, ``pro``, ``max``, or ``unknown``.
        """
        response = self._http.get(ENDPOINT_AUTH_SESSION, rate_limited=False)

        return AccountSession.model_validate(response.json())

    def get_account_settings(self) -> AccountSettings:
        """Return typed user settings for the current token."""
        response = self._http.get(ENDPOINT_USER_SETTINGS, rate_limited=False)

        return AccountSettings.model_validate(response.json())

    def get_account_profile(self) -> AccountProfile:
        """Return combined account data, using settings when session tier is incomplete."""
        provider = getattr(self, "_account_profile_provider", None)
        if provider is None:
            provider = AccountProfileProvider(self.get_account_session, self.get_account_settings)
            self._account_profile_provider = provider

        return provider()

    def close(self) -> None:
        """Close the HTTP session and release all underlying resources."""
        self._http.close()

    def __enter__(self) -> Perplexity:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
