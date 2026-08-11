"""TTL-based conversation cache for the API server."""

from __future__ import annotations

from asyncio import Lock
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from perplexity_webui_scraper.core.conversation import Conversation

_CONVERSATION_TTL_SECONDS: float = 30 * 60


@dataclass
class _CachedConversation:
    """A cached Conversation with continuation metadata and TTL tracking."""

    conversation: Conversation
    last_access: float = field(default_factory=time)
    pending_tool_calls: tuple[dict[str, str], ...] | None = None
    config_fingerprint: str | None = None


class ConversationCache:
    """Async-safe TTL cache for Conversation objects.

    Conversations are keyed by ``(session_token, thread_uuid)`` tuples.
    Stale entries are evicted before each lookup or store.

    Args:
        ttl_seconds: Inactivity timeout in seconds (default: 30 min).
    """

    def __init__(
        self,
        ttl_seconds: float = _CONVERSATION_TTL_SECONDS,
        on_cache: Callable[[str], None] | None = None,
        on_evict: Callable[[str], None] | None = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._on_cache = on_cache
        self._on_evict = on_evict
        self._store: dict[tuple[str, str], _CachedConversation] = {}
        self.lock: Lock = Lock()

    def get(self, token: str, thread_uuid: str) -> Conversation | None:
        """Look up a conversation.  Must be called while ``self.lock`` is held."""
        cached = self.get_entry(token, thread_uuid)
        return cached.conversation if cached is not None else None

    def get_entry(self, token: str, thread_uuid: str) -> _CachedConversation | None:
        """Look up full continuation metadata while lock is held."""
        self._evict_stale()
        cached = self._store.get((token, thread_uuid))

        if cached is None:
            return None

        cached.last_access = time()
        return cached

    def set(
        self,
        token: str,
        thread_uuid: str,
        conversation: Conversation,
        pending_tool_calls: tuple[dict[str, str], ...] | None = None,
        config_fingerprint: str | None = None,
    ) -> None:
        """Store or update conversation and exact continuation metadata."""
        key = (token, thread_uuid)
        existing = self._store.get(key)

        if existing is not None:
            existing.conversation = conversation
            existing.last_access = time()
            existing.pending_tool_calls = pending_tool_calls
            existing.config_fingerprint = config_fingerprint
        else:
            self._store[key] = _CachedConversation(
                conversation=conversation,
                pending_tool_calls=pending_tool_calls,
                config_fingerprint=config_fingerprint,
            )
            if self._on_cache is not None:
                self._on_cache(token)

        self._evict_stale()

    def _evict_stale(self) -> None:
        """Remove all entries exceeding the TTL."""
        now = time()
        stale = [k for k, v in self._store.items() if now - v.last_access > self._ttl]

        for key in stale:
            del self._store[key]
            if self._on_evict is not None:
                self._on_evict(key[0])
