from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from perplexity_webui_scraper.api.app import app
from perplexity_webui_scraper.core import Conversation


TOKEN = "test-token-12345"
AUTH_HEADER = {"Authorization": f"Bearer {TOKEN}"}


def _make_conversation(answer: str, uuid: str = "1234-5678-uuid") -> MagicMock:
    conv = MagicMock(spec=Conversation)
    conv.uuid = uuid
    conv.answer = answer
    return conv


def _make_provider(conversation: MagicMock) -> MagicMock:
    provider = MagicMock()
    provider.create_conversation.return_value = conversation
    return provider


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_messages_api_non_streaming_success(client: TestClient) -> None:
    conv = _make_conversation("Hello from Claude Messages API!")
    provider = _make_provider(conv)

    with patch("perplexity_webui_scraper.api.routes.messages._client_pool.get_or_create", return_value=provider):
        response = client.post(
            "/v1/messages",
            headers=AUTH_HEADER,
            json={
                "model": "anthropic/claude-sonnet-5",
                "messages": [{"role": "user", "content": "Hi"}],
                "system": "You are helpful assistant.",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["stop_reason"] == "end_turn"
    assert len(data["content"]) == 1
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "Hello from Claude Messages API!"


def test_messages_api_streaming_success(client: TestClient) -> None:
    conv = _make_conversation("Streamed message")
    conv.__iter__ = MagicMock(
        return_value=iter(
            [
                SimpleNamespace(chunks=["Streamed "], last_chunk="Streamed ", answer=None),
                SimpleNamespace(chunks=["Streamed ", "message"], last_chunk="message", answer="Streamed message"),
            ]
        )
    )
    provider = _make_provider(conv)

    with patch("perplexity_webui_scraper.api.routes.messages._client_pool.get_or_create", return_value=provider):
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": TOKEN},
            json={
                "model": "anthropic/claude-sonnet-5",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    text = response.text
    assert "event: message_start" in text
    assert "event: content_block_start" in text
    assert "event: content_block_delta" in text
    assert "event: content_block_stop" in text
    assert "event: message_delta" in text
    assert "event: message_stop" in text
