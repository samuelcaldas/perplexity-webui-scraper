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


@pytest.mark.parametrize("endpoint", ["/v1/responses", "/v1/response"])
def test_responses_api_non_streaming_success(client: TestClient, endpoint: str) -> None:
    conv = _make_conversation("Hello from responses API!")
    provider = _make_provider(conv)

    with patch("perplexity_webui_scraper.api.routes.responses._client_pool.get_or_create", return_value=provider):
        response = client.post(
            endpoint,
            headers=AUTH_HEADER,
            json={
                "model": "perplexity/best",
                "input": "Say hello",
                "instructions": "Be concise",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert len(data["output"]) == 1
    assert data["output"][0]["type"] == "message"
    assert data["output"][0]["content"][0]["text"] == "Hello from responses API!"


@pytest.mark.parametrize("endpoint", ["/v1/responses", "/v1/response"])
def test_responses_api_streaming_success(client: TestClient, endpoint: str) -> None:
    conv = _make_conversation("Streamed answer")
    conv.__iter__ = MagicMock(
        return_value=iter(
            [
                SimpleNamespace(chunks=["Streamed "], last_chunk="Streamed ", answer=None),
                SimpleNamespace(chunks=["Streamed ", "answer"], last_chunk="answer", answer="Streamed answer"),
            ]
        )
    )
    provider = _make_provider(conv)

    with patch("perplexity_webui_scraper.api.routes.responses._client_pool.get_or_create", return_value=provider):
        response = client.post(
            endpoint,
            headers=AUTH_HEADER,
            json={
                "model": "perplexity/best",
                "input": "Stream test",
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    text = response.text
    assert "event: response.created" in text
    assert "event: response.text.delta" in text
    assert "event: response.done" in text
    assert "data: [DONE]" in text
