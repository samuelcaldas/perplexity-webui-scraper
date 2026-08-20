"""Comprehensive test suite testing EVERY model in the /models endpoint catalog across all APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from perplexity_webui_scraper import ModelAccessError, ModelRiskWarning, ModelStatusError
from perplexity_webui_scraper.api.app import app
from perplexity_webui_scraper.config.conversation import ConversationConfig
from perplexity_webui_scraper.core import Conversation
from perplexity_webui_scraper.core.account import AccountProfile, AccountSession, AccountUser, ensure_model_access
from perplexity_webui_scraper.core.payload import build_payload
from perplexity_webui_scraper.models.registry import MODELS


if TYPE_CHECKING:
    from perplexity_webui_scraper.models.types import Model


TOKEN = "test-models-token-xyz789"
AUTH_HEADER = {"Authorization": f"Bearer {TOKEN}"}

ALL_MODELS: list[Model] = MODELS.list_all()
ALL_MODEL_IDS: list[str] = [m.id for m in ALL_MODELS]
AVAILABLE_MODELS: list[Model] = [m for m in ALL_MODELS if m.status == "available"]
AVAILABLE_MODEL_IDS: list[str] = [m.id for m in AVAILABLE_MODELS]
UNAVAILABLE_MODELS: list[Model] = [m for m in ALL_MODELS if m.status == "unavailable"]
UNAVAILABLE_MODEL_IDS: list[str] = [m.id for m in UNAVAILABLE_MODELS]
THINKING_MODELS: list[Model] = [m for m in ALL_MODELS if m.supports_thinking]
THINKING_MODEL_IDS: list[str] = [m.id for m in THINKING_MODELS]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_mock_conversation(answer: str = "Mock answer for model test") -> MagicMock:
    conv = MagicMock(spec=Conversation)
    conv.uuid = "conv-uuid-mock-1234"
    conv.answer = answer
    return conv


# ---------------------------------------------------------------------------
# 1. Endpoint Schema & Parity for /models and /v1/models (Every Model in Catalog)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint", ["/models", "/v1/models"])
def test_models_endpoint_lists_every_model_in_catalog(client: TestClient, endpoint: str) -> None:
    """Verify that both /models and /v1/models return all 46 models with complete metadata."""
    response = client.get(endpoint)
    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "list"
    items = data["data"]
    assert len(items) == len(ALL_MODELS) == 46

    # Verify ID ordering and uniqueness
    returned_ids = [item["id"] for item in items]
    assert returned_ids == ALL_MODEL_IDS
    assert len(returned_ids) == len(set(returned_ids))

    # Verify each model's full metadata
    expected_by_id = {m.id: m for m in ALL_MODELS}
    for item in items:
        model = expected_by_id[item["id"]]
        assert item["object"] == "model"
        assert item["created"] == 0
        assert item["owned_by"] == model.provider

        meta = item["perplexity"]
        assert meta["min_tier"] == model.min_tier
        assert meta["is_official"] == model.is_official
        assert meta["status"] == model.status
        expected_tested_at = model.last_tested_at.isoformat().replace("+00:00", "Z") if model.last_tested_at else None
        assert meta["last_tested_at"] == expected_tested_at


# ---------------------------------------------------------------------------
# 2. Retrieve Single Model by ID and Alias (Every Model in Catalog)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.id)
@pytest.mark.parametrize("prefix", ["/models", "/v1/models"])
def test_retrieve_single_model_by_canonical_id(client: TestClient, model: Model, prefix: str) -> None:
    """Verify retrieving every single model by its canonical ID."""
    response = client.get(f"{prefix}/{model.id}")
    assert response.status_code == 200
    item = response.json()

    assert item["id"] == model.id
    assert item["object"] == "model"
    assert item["created"] == 0
    assert item["owned_by"] == model.provider
    assert item["perplexity"]["status"] == model.status
    assert item["perplexity"]["min_tier"] == model.min_tier


@pytest.mark.parametrize(
    ("alias", "expected_canonical_id"),
    [
        ("gpt-5.6-terra", "openai/gpt-5.6-terra"),
        ("gpt-5.6-sol", "openai/gpt-5.6-sol"),
        ("claude-sonnet-5", "anthropic/claude-sonnet-5"),
        ("sonar-2", "perplexity/sonar-2"),
        ("deep-research", "perplexity/deep-research"),
        ("best", "perplexity/best"),
        ("glm-5.2", "z-ai/glm-5.2"),
        ("kimi-k3", "moonshot/kimi-k3"),
        ("grok-4.5", "x-ai/grok-4.5"),
        ("nemotron-3-ultra", "nvidia/nemotron-3-ultra"),
        ("claude-opus-5", "anthropic/claude-opus-5"),
        ("claude-opus-4.8", "anthropic/claude-opus-4.8"),
        ("kimi-k2.6", "moonshot/kimi-k2.6"),
        ("nemotron-3-super", "nvidia/nemotron-3-super"),
    ],
)
def test_retrieve_single_model_by_alias(client: TestClient, alias: str, expected_canonical_id: str) -> None:
    """Verify single model retrieval using bare/short aliases."""
    response = client.get(f"/v1/models/{alias}")
    assert response.status_code == 200
    item = response.json()
    assert item["id"] == expected_canonical_id


def test_retrieve_non_existent_model_returns_404(client: TestClient) -> None:
    """Verify 404 response for unknown models."""
    response = client.get("/v1/models/non-existent-provider/fake-model-xyz")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "model_not_found"
    assert "does not exist" in data["error"]["message"]


# ---------------------------------------------------------------------------
# 3. Model Resolution & Identifier Mapping (Every Model in Catalog)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.id)
def test_model_registry_resolves_every_model(model: Model) -> None:
    """Verify registry resolves every canonical model and preserves identifiers."""
    resolved = MODELS.resolve(model.id)
    assert resolved.id == model.id
    assert resolved.provider == model.provider
    assert resolved.mode == model.mode
    assert resolved.min_tier == model.min_tier
    assert resolved.status == model.status
    assert resolved.identifier == model.identifier

    # Test resolve_for_use
    if model.status == "available":
        usable = MODELS.resolve_for_use(model.id)
        assert usable.id == model.id
    else:
        with pytest.raises(ModelStatusError):
            MODELS.resolve_for_use(model.id)

        with pytest.warns(ModelRiskWarning):
            usable_risky = MODELS.resolve_for_use(model.id, allow_risky_model=True)
            assert usable_risky.id == model.id


# ---------------------------------------------------------------------------
# 4. Thinking & Reasoning Effort Resolution (Every Thinking-Capable Model)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", THINKING_MODELS, ids=lambda m: m.id)
def test_thinking_capable_models_resolution(model: Model) -> None:
    """Verify thinking identifier switching for all 37 thinking models."""
    # 1. Explicit thinking=True
    with pytest.warns(ModelRiskWarning) if model.status != "available" else pytest.MonkeyPatch.context() as _:
        resolved_think = MODELS.resolve_for_use(model.id, allow_risky_model=True, thinking=True)
        if model.thinking_identifier:
            assert resolved_think.identifier == model.thinking_identifier
        else:
            assert resolved_think.identifier == model.identifier

    # 2. Explicit reasoning_effort="high"
    with pytest.warns(ModelRiskWarning) if model.status != "available" else pytest.MonkeyPatch.context() as _:
        resolved_effort = MODELS.resolve_for_use(model.id, allow_risky_model=True, reasoning_effort="high")
        if model.thinking_identifier:
            assert resolved_effort.identifier == model.thinking_identifier
        else:
            assert resolved_effort.identifier == model.identifier

    # 3. Explicit thinking=False (instant mode)
    with pytest.warns(ModelRiskWarning) if model.status != "available" else pytest.MonkeyPatch.context() as _:
        resolved_instant = MODELS.resolve_for_use(model.id, allow_risky_model=True, thinking=False)
        if not model.thinking_only:
            assert resolved_instant.identifier == model.identifier


# ---------------------------------------------------------------------------
# 5. Account Tier Entitlement Across Catalog Models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "expected_available_count"),
    [
        ("free", 1),  # Only perplexity/best
        ("pro", 12),  # Pro & free available models (excludes max-tier models: gpt-5.6-sol, opus-5, opus-4.8)
        ("max", 15),  # All 15 available models
    ],
)
def test_authenticated_models_list_filters_by_tier(
    client: TestClient, tier: str, expected_available_count: int
) -> None:
    """Verify GET /models filters available models according to account tier."""
    fake_client = MagicMock()
    session = AccountSession(user=AccountUser(subscription_tier=tier))
    fake_client.get_account_profile.return_value = AccountProfile(session=session)

    with patch("perplexity_webui_scraper.api.routes.models.client_pool.get_or_create", return_value=fake_client):
        response = client.get("/v1/models", headers=AUTH_HEADER)
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data) == expected_available_count
        assert all(item["perplexity"]["status"] == "available" for item in data)


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.id)
def test_ensure_model_access_rules_across_all_models(model: Model) -> None:
    """Verify tier gating validation function across all models."""
    session_max = AccountSession.model_validate({"user": {"subscription_tier": "max"}})
    # Max tier has access to any model tier definition
    ensure_model_access(session_max, model)

    if model.min_tier == "max":
        session_pro = AccountSession.model_validate({"user": {"subscription_tier": "pro"}})
        with pytest.raises(ModelAccessError):
            ensure_model_access(session_pro, model)


# ---------------------------------------------------------------------------
# 6. API Invocation Across EVERY Model in Catalog (/v1/chat/completions)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.id)
def test_chat_completions_for_every_model_in_catalog(client: TestClient, model: Model) -> None:
    """Verify /v1/chat/completions accepts every model in the catalog."""
    mock_provider = patch("perplexity_webui_scraper.api.routes.completions._client_pool.get_or_create").start()
    conv = _make_mock_conversation(f"Answer for {model.id}")
    mock_provider.return_value.create_conversation.return_value = conv

    try:
        messages = [{"role": "user", "content": "Hello"}]

        if model.status == "available":
            # Available models work directly
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADER,
                json={"model": model.id, "messages": messages},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == f"Answer for {model.id}"
        else:
            # Unavailable models require explicit allow_risky_model
            denied_resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADER,
                json={"model": model.id, "messages": messages},
            )
            assert denied_resp.status_code == 400
            assert denied_resp.json()["error"]["code"] == "model_status_confirmation_required"

            allowed_resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADER,
                json={
                    "model": model.id,
                    "messages": messages,
                    "perplexity": {"allow_risky_model": True},
                },
            )
            assert allowed_resp.status_code == 200
            data = allowed_resp.json()
            assert data["choices"][0]["message"]["content"] == f"Answer for {model.id}"

    finally:
        patch.stopall()


# ---------------------------------------------------------------------------
# 7. Responses API Invocation Across EVERY Model in Catalog (/v1/responses)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.id)
def test_responses_api_for_every_model_in_catalog(client: TestClient, model: Model) -> None:
    """Verify /v1/responses accepts every model in the catalog."""
    mock_provider = patch("perplexity_webui_scraper.api.routes.responses._client_pool.get_or_create").start()
    conv = _make_mock_conversation(f"Responses API answer for {model.id}")
    mock_provider.return_value.create_conversation.return_value = conv

    try:
        payload: dict = {
            "model": model.id,
            "input": "Summarize status.",
        }
        if model.status != "available":
            payload["perplexity"] = {"allow_risky_model": True}

        resp = client.post(
            "/v1/responses",
            headers=AUTH_HEADER,
            json=payload,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["output"][0]["content"][0]["text"] == f"Responses API answer for {model.id}"

    finally:
        patch.stopall()


# ---------------------------------------------------------------------------
# 8. Messages API Invocation Across EVERY Model in Catalog (/v1/messages)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.id)
def test_messages_api_for_every_model_in_catalog(client: TestClient, model: Model) -> None:
    """Verify /v1/messages accepts every model in the catalog."""
    mock_provider = patch("perplexity_webui_scraper.api.routes.messages._client_pool.get_or_create").start()
    conv = _make_mock_conversation(f"Anthropic API answer for {model.id}")
    mock_provider.return_value.create_conversation.return_value = conv

    try:
        payload: dict = {
            "model": model.id,
            "messages": [{"role": "user", "content": "Hello via Messages API"}],
        }
        if model.status != "available":
            payload["perplexity"] = {"allow_risky_model": True}

        resp = client.post(
            "/v1/messages",
            headers={"x-api-key": TOKEN},
            json=payload,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["content"][0]["text"] == f"Anthropic API answer for {model.id}"

    finally:
        patch.stopall()


# ---------------------------------------------------------------------------
# 9. Payload Construction for Every Model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.id)
def test_payload_conversation_config_for_every_model(model: Model) -> None:
    """Verify build_payload constructs expected model_preference and mode for each model."""
    config = ConversationConfig(model=model.id, allow_risky_model=True)
    payload = build_payload(
        query="test query",
        model=model,
        file_urls=[],
        config=config,
        backend_uuid=None,
        read_write_token=None,
    )
    params = payload["params"]
    assert params["model_preference"] == model.identifier
    assert params["mode"] == model.mode
