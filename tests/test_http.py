from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from curl_cffi.requests import exceptions as curl_exceptions
from pytest import raises

from perplexity_webui_scraper._internal.exceptions import RateLimitError, TransientHTTPError
from perplexity_webui_scraper.http.client import HTTPClient


class _HTTPStatusError(Exception):
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


class _FakeResponse:
    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        self.status_code = status_code
        self.url = "https://www.perplexity.ai/rest/test"
        self.text = "rate limited" if status_code == 429 else "ok"
        self.headers = {"Retry-After": retry_after} if retry_after is not None else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _HTTPStatusError(self)


class _FakeSession:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        self.calls += 1

        return _FakeResponse(200)

    def post(self, url: str, json: dict[str, Any] | None = None, stream: bool = False) -> _FakeResponse:
        self.calls += 1

        return _FakeResponse(429 if self.calls == 1 else 200)

    def close(self) -> None:
        return None


class _StreamingFailureResponse(_FakeResponse):
    def __init__(self) -> None:
        super().__init__(200)
        self.closed = False

    def iter_lines(self) -> list[bytes]:
        raise curl_exceptions.ConnectionError("connection dropped")

    def close(self) -> None:
        self.closed = True


class _StreamingFailureSession:
    def __init__(self, response: _StreamingFailureResponse) -> None:
        self.response = response

    def post(self, url: str, json: dict[str, Any] | None = None, stream: bool = False) -> _StreamingFailureResponse:
        return self.response

    def close(self) -> None:
        return None


class _FailingSession:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, url: str, json: dict[str, Any] | None = None, stream: bool = False) -> _FakeResponse:
        self.calls += 1
        return _FakeResponse(503)

    def close(self) -> None:
        return None


def test_http_client_retries_translated_rate_limit_error() -> None:
    client = HTTPClient(
        "token",
        max_retries=1,
        retry_base_delay=0,
        retry_jitter=0,
        rotate_fingerprint=False,
        requests_per_second=0,
    )
    client.close()
    fake_session = _FakeSession()
    client._session = cast("Any", fake_session)

    response = client.post("/rest/test", json={"query": "hello"})

    assert response.status_code == 200
    assert fake_session.calls == 2


def test_http_get_can_skip_rate_limiter() -> None:
    client = HTTPClient(
        "token",
        max_retries=0,
        retry_base_delay=0,
        retry_jitter=0,
        rotate_fingerprint=False,
        requests_per_second=1,
    )
    client.close()
    fake_session = _FakeSession()
    client._session = cast("Any", fake_session)
    client._rate_limiter = cast("Any", None)

    response = client.get("/api/auth/session", rate_limited=False)

    assert response.status_code == 200
    assert fake_session.calls == 1


def test_http_client_preserves_numeric_retry_after_metadata() -> None:
    client = HTTPClient("token", max_retries=0, rotate_fingerprint=False, requests_per_second=0)
    client.close()
    response = _FakeResponse(429, retry_after="7")
    session = MagicMock()
    session.post.return_value = response
    client._session = cast("Any", session)

    with raises(RateLimitError) as raised:
        client.post("/rest/test", json={"query": "hello"})

    assert raised.value.retry_after == 7.0
    assert raised.value.url == "https://www.perplexity.ai/rest/test"


def test_http_client_fails_fast_after_transient_circuit_opens() -> None:
    client = HTTPClient(
        "token",
        max_retries=0,
        retry_base_delay=0,
        retry_jitter=0,
        circuit_failure_threshold=1,
        circuit_cooldown=60,
        rotate_fingerprint=False,
        requests_per_second=1000,
    )
    client.close()
    session = _FailingSession()
    client._session = cast("Any", session)

    with raises(TransientHTTPError):
        client.post("/rest/test", json={"query": "first"})
    with raises(RateLimitError) as raised:
        client.post("/rest/test", json={"query": "second"})

    assert raised.value.retry_after is not None
    assert session.calls == 1


def test_stream_transport_error_is_typed_and_closes_response() -> None:
    client = HTTPClient("token", max_retries=0, rotate_fingerprint=False, requests_per_second=0)
    client.close()
    response = _StreamingFailureResponse()
    client._session = cast("Any", _StreamingFailureSession(response))

    with raises(TransientHTTPError):
        list(client.stream_ask({"query": "hello"}))

    assert response.closed
