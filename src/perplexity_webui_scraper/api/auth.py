"""Authentication helpers for the API server.

``extract_token()`` validates the ``Authorization: Bearer`` header.
``ClientPool`` maintains a per-token cache of ``Perplexity`` client instances
to avoid recreating HTTP sessions on every request.
"""

from __future__ import annotations

from asyncio import Lock
from collections import OrderedDict
from threading import RLock

from fastapi import HTTPException

from perplexity_webui_scraper import Perplexity
from perplexity_webui_scraper._internal.constants import AUTH_BEARER_PREFIX
from perplexity_webui_scraper.config.client import ClientConfig
from perplexity_webui_scraper.http.resilience import RateLimiter


def extract_token(authorization: str | None) -> str:
    """Extract the raw session token from the ``Authorization: Bearer`` header.

    Args:
        authorization: Raw value of the ``Authorization`` header.

    Returns:
        The session token string (everything after ``"Bearer "``).

    Raises:
        HTTPException: 401 if the header is missing, not ``Bearer``, or empty.
    """
    if not authorization or not authorization.startswith(AUTH_BEARER_PREFIX):
        raise HTTPException(
            status_code=401,
            detail=(
                "Missing or invalid Authorization header. "
                "Pass your Perplexity session token as: "
                "Authorization: Bearer <token>"
            ),
        )

    token = authorization[len(AUTH_BEARER_PREFIX) :]

    if not token.strip():
        raise HTTPException(status_code=401, detail="Bearer token is empty.")

    return token.strip()


class ClientPool:
    """Bounded LRU cache of per-token Perplexity clients."""

    def __init__(self, max_size: int = 128) -> None:
        """Create client pool with bounded cached session count."""
        if max_size < 1:
            raise ValueError("max_size must be positive")

        self._max_size = max_size
        self._clients: OrderedDict[str, Perplexity] = OrderedDict()
        self._transient_clients: dict[str, Perplexity] = {}
        self._client_cached_refs: dict[str, int] = {}
        self._pending_discards: set[str] = set()
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._request_locks: OrderedDict[str, Lock] = OrderedDict()
        self._request_users: dict[str, int] = {}
        self._state_lock = RLock()

    def get_request_lock(self, token: str) -> Lock:
        """Return process-local lock and retain its entry until request release."""
        with self._state_lock:
            lock = self._request_locks.pop(token, None)
            if lock is None:
                lock = Lock()
            self._request_locks[token] = lock
            self._request_users[token] = self._request_users.get(token, 0) + 1
            self._evict_idle_locks()
            return lock

    def release_request_lock(self, token: str, lock: Lock) -> None:
        """Release request ownership and clean idle client/lock entries."""
        with self._state_lock:
            if self._request_locks.get(token) is not lock:
                return

            users = self._request_users.get(token, 0)
            if users <= 1:
                self._request_users.pop(token, None)
            else:
                self._request_users[token] = users - 1

            self._cleanup_client(token)
            self._evict_idle_locks()

    def pin(self, token: str) -> None:
        """Protect client for token while cached conversation references it."""
        with self._state_lock:
            self._client_cached_refs[token] = self._client_cached_refs.get(token, 0) + 1

    def unpin(self, token: str) -> None:
        """Release one cached conversation reference for token."""
        with self._state_lock:
            refs = self._client_cached_refs.get(token, 0)
            if refs <= 1:
                self._client_cached_refs.pop(token, None)
            else:
                self._client_cached_refs[token] = refs - 1
            self._cleanup_client(token)

    def get_or_create(self, token: str) -> Perplexity:
        """Return cached client or create one for *token*."""
        with self._state_lock:
            client = self._clients.pop(token, None)
            if client is not None:
                self._clients[token] = client
                return client

            client = self._transient_clients.get(token)
            if client is not None:
                return client

            config = ClientConfig()
            rate_limiter = self._rate_limiters.setdefault(
                token,
                RateLimiter(
                    requests_per_second=config.requests_per_second,
                    circuit_failure_threshold=config.circuit_failure_threshold,
                    circuit_cooldown=config.circuit_cooldown,
                ),
            )
            client = Perplexity(token, config=config, rate_limiter=rate_limiter)
            if self._can_cache_client(token):
                self._clients[token] = client
            else:
                self._transient_clients[token] = client
            return client

    def _can_cache_client(self, token: str) -> bool:
        """Make room for a new client without evicting protected clients."""
        if len(self._clients) < self._max_size:
            return True

        evictable = next((key for key in self._clients if not self._is_protected(key)), None)
        if evictable is None:
            return False

        self._close_client(evictable)
        return True

    def discard(self, token: str, client: Perplexity | None = None) -> None:
        """Remove client after provider failure without closing active references."""
        with self._state_lock:
            current_client = self._clients.get(token) or self._transient_clients.get(token)
            if current_client is None:
                if client is not None and not self._is_protected(token):
                    client.close()
                return

            if client is not None and current_client is not client:
                if not self._is_protected(token):
                    client.close()
                return

            self._pending_discards.add(token)
            self._cleanup_client(token)

    def _evict_excess(self) -> None:
        """Close idle least-recently-used clients above configured bound."""
        with self._state_lock:
            while len(self._clients) > self._max_size:
                token = next((key for key in self._clients if not self._is_protected(key)), None)
                if token is None:
                    return
                self._close_client(token)

    def _evict_idle_locks(self) -> None:
        """Bound retained idle request locks without touching active locks."""
        with self._state_lock:
            while len(self._request_locks) > self._max_size:
                token = next((key for key in self._request_locks if not self._request_users.get(key)), None)
                if token is None:
                    return
                self._request_locks.pop(token, None)

    def _cleanup_client(self, token: str) -> None:
        """Close token client once no request or cache references remain."""
        with self._state_lock:
            if self._is_protected(token):
                return

            if token in self._transient_clients:
                self._close_transient_client(token)
                return

            if token not in self._clients:
                return
            if token in self._pending_discards or len(self._clients) > self._max_size:
                self._close_client(token)
                self._evict_excess()

    def _is_protected(self, token: str) -> bool:
        """Return whether active work or cached continuation needs token client."""
        with self._state_lock:
            return bool(self._request_users.get(token) or self._client_cached_refs.get(token))

    def _close_client(self, token: str) -> None:
        """Remove and close one unprotected cached client."""
        with self._state_lock:
            client = self._clients.pop(token, None)
            self._pending_discards.discard(token)
            self._rate_limiters.pop(token, None)
            if client is not None:
                client.close()

    def _close_transient_client(self, token: str) -> None:
        """Close one unprotected client kept outside bounded cache."""
        with self._state_lock:
            client = self._transient_clients.pop(token, None)
            self._pending_discards.discard(token)
            self._rate_limiters.pop(token, None)
            if client is not None:
                client.close()


client_pool = ClientPool()
