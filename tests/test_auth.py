from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

from pytest import raises

from perplexity_webui_scraper.api.auth import ClientPool
from perplexity_webui_scraper.api.routes.completions import chat_completions
from perplexity_webui_scraper.api.schemas.request import ChatCompletionRequest


if TYPE_CHECKING:
    from fastapi import Request


def test_client_pool_closes_least_recently_used_client_when_full() -> None:
    first_client = MagicMock()
    second_client = MagicMock()
    third_client = MagicMock()

    with patch(
        "perplexity_webui_scraper.api.auth.Perplexity",
        side_effect=[first_client, second_client, third_client],
    ):
        pool = ClientPool(max_size=2)
        assert pool.get_or_create("first") is first_client
        assert pool.get_or_create("second") is second_client
        assert pool.get_or_create("third") is third_client

    first_client.close.assert_called_once_with()
    second_client.close.assert_not_called()
    third_client.close.assert_not_called()


def test_client_pool_discards_and_closes_matching_client() -> None:
    client = MagicMock()
    replacement_client = MagicMock()

    with patch(
        "perplexity_webui_scraper.api.auth.Perplexity",
        side_effect=[client, replacement_client],
    ):
        pool = ClientPool()
        cached_client = pool.get_or_create("token")
        pool.discard("token", cached_client)
        assert pool.get_or_create("token") is replacement_client

    client.close.assert_called_once_with()


def test_client_pool_serializes_concurrent_client_creation() -> None:
    created_clients: list[MagicMock] = []

    def create_client(*_args: object, **_kwargs: object) -> MagicMock:
        client = MagicMock()
        created_clients.append(client)
        sleep(0.01)
        return client

    with patch("perplexity_webui_scraper.api.auth.Perplexity", side_effect=create_client):
        pool = ClientPool(max_size=4)
        with ThreadPoolExecutor(max_workers=8) as executor:
            clients = list(executor.map(lambda _index: pool.get_or_create("same-token"), range(16)))

    assert len(created_clients) == 1
    assert all(client is created_clients[0] for client in clients)


def test_client_pool_bounds_idle_request_locks() -> None:
    pool = ClientPool(max_size=2)

    for token in ("first", "second", "third"):
        lock = pool.get_request_lock(token)
        pool.release_request_lock(token, lock)

    assert len(pool._request_locks) == 2


def test_client_pool_keeps_cached_bound_when_all_clients_are_active() -> None:
    first_client = MagicMock()
    second_client = MagicMock()

    with patch(
        "perplexity_webui_scraper.api.auth.Perplexity",
        side_effect=[first_client, second_client],
    ):
        pool = ClientPool(max_size=1)
        first_lock = pool.get_request_lock("first")
        second_lock = pool.get_request_lock("second")
        assert pool.get_or_create("first") is first_client
        assert pool.get_or_create("second") is second_client

        assert len(pool._clients) == 1
        first_client.close.assert_not_called()
        second_client.close.assert_not_called()

        pool.release_request_lock("first", first_lock)
        pool.release_request_lock("second", second_lock)

    first_client.close.assert_not_called()
    second_client.close.assert_called_once_with()


def test_client_pool_does_not_close_client_during_active_request() -> None:
    first_client = MagicMock()
    second_client = MagicMock()

    with patch(
        "perplexity_webui_scraper.api.auth.Perplexity",
        side_effect=[first_client, second_client],
    ):
        pool = ClientPool(max_size=1)
        first_lock = pool.get_request_lock("first")
        assert pool.get_or_create("first") is first_client
        assert pool.get_or_create("second") is second_client
        first_client.close.assert_not_called()

        pool.release_request_lock("first", first_lock)

    first_client.close.assert_not_called()


def test_client_pool_does_not_close_client_while_cached() -> None:
    first_client = MagicMock()
    second_client = MagicMock()

    with patch(
        "perplexity_webui_scraper.api.auth.Perplexity",
        side_effect=[first_client, second_client],
    ):
        pool = ClientPool(max_size=1)
        assert pool.get_or_create("first") is first_client
        pool.pin("first")
        assert pool.get_or_create("second") is second_client
        first_client.close.assert_not_called()

        pool.unpin("first")

    first_client.close.assert_not_called()


def test_cancelled_request_releases_client_pool_lock_reference() -> None:
    token = "cancelled-token"
    request = ChatCompletionRequest.model_validate(
        {"model": "perplexity/best", "messages": [{"role": "user", "content": "Hello"}]}
    )
    pool = ClientPool(max_size=2)

    async def parse_request(_raw_request: object) -> ChatCompletionRequest:
        return request

    async def scenario() -> None:
        active_lock = pool.get_request_lock(token)
        await active_lock.acquire()

        with (
            patch("perplexity_webui_scraper.api.routes.completions._client_pool", pool),
            patch("perplexity_webui_scraper.api.routes.completions._parse_request", parse_request),
            patch("perplexity_webui_scraper.api.routes.completions._validate_model"),
        ):
            task = asyncio.create_task(chat_completions(cast("Request", object()), f"Bearer {token}"))
            for _ in range(100):
                if pool._request_users.get(token) == 2:
                    break
                await asyncio.sleep(0)
            else:
                raise AssertionError("request did not start waiting on token lock")

            task.cancel()
            with raises(asyncio.CancelledError):
                await task

        assert pool._request_users[token] == 1
        active_lock.release()
        pool.release_request_lock(token, active_lock)

    asyncio.run(scenario())
