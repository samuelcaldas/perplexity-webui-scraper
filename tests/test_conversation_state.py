from __future__ import annotations

from typing import Any, cast

from pytest import raises

from perplexity_webui_scraper._internal.constants import ENDPOINT_AUTH_SESSION
from perplexity_webui_scraper._internal.exceptions import StreamingError
from perplexity_webui_scraper.config.conversation import ConversationConfig
from perplexity_webui_scraper.core.conversation import Conversation


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _SequenceHTTP:
    def __init__(self, streams: list[list[bytes]]) -> None:
        self._streams = streams
        self._stream_index = 0

    def get(self, endpoint: str, rate_limited: bool = True) -> _Response:
        assert endpoint == ENDPOINT_AUTH_SESSION
        assert rate_limited is False
        return _Response({"user": {"subscription_tier": "pro"}})

    def init_search(self, _query: str) -> None:
        return None

    def stream_ask(self, _payload: dict[str, Any]):
        stream = self._streams[self._stream_index]
        self._stream_index += 1
        yield from stream


def _conversation(http: _SequenceHTTP) -> Conversation:
    return Conversation(cast("Any", http), ConversationConfig(model="perplexity/best"))


def _successful_stream(uuid: str, answer: str) -> list[bytes]:
    return [
        f'data: {{"backend_uuid":"{uuid}","text":"{{\\"answer\\":\\"{answer}\\"}}"}}\n'.encode(),
        b'data: {"final":true}\n',
    ]


def test_direct_conversation_retry_restores_previous_completed_state() -> None:
    http = _SequenceHTTP(
        [
            _successful_stream("stable-thread", "stable answer"),
            [b'data: {"backend_uuid":"partial-thread","text":"{\\"answer\\":\\"partial\\"}"}\n'],
        ]
    )
    conversation = _conversation(http)

    conversation.ask("first")

    with raises(StreamingError):
        conversation.ask("second")

    assert conversation.answer == "stable answer"
    assert conversation.uuid == "stable-thread"


def test_failed_stream_restores_previous_completed_state() -> None:
    http = _SequenceHTTP(
        [
            _successful_stream("stable-thread", "stable answer"),
            [b'data: {"backend_uuid":"partial-thread","text":"{\\"answer\\":\\"partial\\"}"}\n'],
        ]
    )
    conversation = _conversation(http)
    conversation.ask("first")
    conversation.ask("second", stream=True)

    with raises(StreamingError):
        list(conversation)

    assert conversation.answer == "stable answer"
    assert conversation.uuid == "stable-thread"


def test_successful_stream_commits_new_conversation_state() -> None:
    http = _SequenceHTTP([_successful_stream("stream-thread", "stream answer")])
    conversation = _conversation(http)

    conversation.ask("stream", stream=True)
    list(conversation)

    assert conversation.answer == "stream answer"
    assert conversation.uuid == "stream-thread"
