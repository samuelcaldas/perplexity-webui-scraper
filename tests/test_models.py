from __future__ import annotations

from copy import deepcopy

from pydantic import ValidationError
from pytest import mark, raises, warns

from perplexity_webui_scraper import ModelRiskWarning, ModelStatusError
from perplexity_webui_scraper.models.registry import MODELS, ModelRegistry, parse_thinking_modifier
from perplexity_webui_scraper.models.types import Model


_MODEL: dict[str, object] = {
    "id": "provider/model",
    "name": "Provider Model",
    "description": "A test model.",
    "identifier": "provider_model",
    "tool_name": "pplx_provider_model",
    "provider": "provider",
    "min_tier": "pro",
    "mode": "copilot",
}


def test_bundled_model_registry_is_valid() -> None:
    models = MODELS.list_all()
    ids = [model.id for model in models]
    tool_names = [model.tool_name for model in models]

    assert models
    assert all(model.status in {"available", "unknown", "unavailable"} for model in models)
    assert all(isinstance(model.is_official, bool) for model in models)
    assert all(model.last_tested_at is not None for model in models if model.status in {"available", "unavailable"})
    official_positions = [index for index, model in enumerate(models) if model.is_official]
    historical_positions = [index for index, model in enumerate(models) if not model.is_official]
    assert not historical_positions or max(official_positions) < min(historical_positions)
    assert len(ids) == len(set(ids))
    assert len(tool_names) == len(set(tool_names))
    assert all(model.id and model.identifier and model.tool_name for model in models)
    assert all(model.provider and model.description for model in models)


def test_model_rejects_unknown_fields() -> None:
    model_data = dict(_MODEL)
    model_data["unexpected"] = True

    with raises(ValidationError):
        Model.model_validate(model_data)


@mark.parametrize("timestamp", ["2026-07-20T23:34:21", "2026-07-21T02:34:21+03:00"])
def test_model_rejects_non_utc_test_timestamps(timestamp: str) -> None:
    with raises(ValidationError, match="last_tested_at must use UTC"):
        Model.model_validate({**_MODEL, "last_tested_at": timestamp})


def test_model_registry_rejects_duplicate_ids() -> None:
    duplicate = deepcopy(_MODEL)
    duplicate["tool_name"] = "pplx_provider_model_other"

    with raises(ValueError, match="Duplicate model id"):
        ModelRegistry([_MODEL, duplicate])


def test_model_registry_rejects_duplicate_tool_names() -> None:
    duplicate = deepcopy(_MODEL)
    duplicate["id"] = "provider/other-model"

    with raises(ValueError, match="Duplicate MCP tool name"):
        ModelRegistry([_MODEL, duplicate])


@mark.parametrize("legacy_field", ["unstable", "disabled", "warning"])
def test_model_rejects_legacy_availability_fields(legacy_field: str) -> None:
    with raises(ValidationError):
        Model.model_validate({**_MODEL, legacy_field: True})


def test_unknown_model_requires_acknowledgement() -> None:
    registry = ModelRegistry([{**_MODEL, "status": "unknown"}])
    with raises(ModelStatusError) as exc_info:
        registry.resolve_for_use("provider/model")
    assert exc_info.value.status == "unknown"

    with warns(ModelRiskWarning):
        model = registry.resolve_for_use("provider/model", allow_risky_model=True)
    assert model.identifier == "provider_model"


@mark.parametrize("status", ["unknown", "unavailable"])
def test_other_risky_statuses_use_the_same_acknowledgement(status: str) -> None:
    registry = ModelRegistry([{**_MODEL, "status": status}])
    with raises(ModelStatusError) as exc_info:
        registry.resolve_for_use("provider/model")
    assert exc_info.value.status == status
    with warns(ModelRiskWarning):
        assert registry.resolve_for_use("provider/model", allow_risky_model=True).status == status


def test_custom_model_is_explicit_and_validated() -> None:
    with raises(ModelStatusError):
        MODELS.resolve_for_use("custom:gpt57")
    with warns(ModelRiskWarning):
        model = MODELS.resolve_for_use(
            "custom:gpt57",
            allow_risky_model=True,
            custom_model_mode="search",
        )
    assert model.identifier == "gpt57"
    assert model.mode == "search"
    assert model.min_tier is None
    assert model.status == "unknown"
    with raises(ValueError, match="Custom model identifiers"):
        MODELS.resolve_for_use("custom:", allow_risky_model=True)
    with raises(ValueError, match="Unknown model"):
        MODELS.resolve_for_use("gpt57", allow_risky_model=True)


def test_dynamic_unregistered_model_fallback() -> None:
    with raises(ModelStatusError):
        MODELS.resolve_for_use("openai/gpt-6.0", allow_unregistered=True)
    with warns(ModelRiskWarning):
        model = MODELS.resolve_for_use(
            "openai/gpt-6.0",
            allow_risky_model=True,
            allow_unregistered=True,
        )
    assert model.id == "openai/gpt-6.0"
    assert model.identifier == "gpt-6.0"
    assert model.provider == "openai"
    assert model.status == "unknown"


@mark.parametrize(
    ("raw", "expected_base", "expected_thinking", "expected_effort"),
    [
        ("claude45haikuthinking", "claude45haiku", True, None),
        ("kimik2thinking", "kimik2", True, None),
        ("anthropic/claude-sonnet-5-thinking", "anthropic/claude-sonnet-5", True, None),
        ("google/gemini-3.1-pro-thinking-high", "google/gemini-3.1-pro", True, "high"),
        ("google/gemini-3.1-pro-thinking-low", "google/gemini-3.1-pro", True, "low"),
        ("openai/gpt51-low-thinking", "openai/gpt51", True, "low"),
        ("openai/gpt51-thinking-high", "openai/gpt51", True, "high"),
        ("claude-3.7-sonnet-thinking-20250219", "claude-3.7-sonnet-20250219", True, None),
        ("x-ai/grok4nonthinking", "x-ai/grok4", False, None),
        ("x-ai/grok41nonreasoning", "x-ai/grok41", False, None),
        ("x-ai/grok41reasoning", "x-ai/grok41", True, None),
        ("openai/gpt-5.6-terra[1m]", "openai/gpt-5.6-terra", None, None),
    ],
)
def test_parse_thinking_modifier(
    raw: str,
    expected_base: str,
    expected_thinking: bool | None,
    expected_effort: str | None,
) -> None:
    base, thinking, effort = parse_thinking_modifier(raw)
    assert base == expected_base
    assert thinking is expected_thinking
    assert effort == expected_effort


def test_thinking_only_model_resolution() -> None:
    # Kimi K3 is thinking_only: true
    model_k3 = MODELS.resolve_for_use("moonshot/kimi-k3")
    assert model_k3.identifier == "kimik3thinking"
    assert model_k3.thinking_only is True

    # Deep research is thinking_only: true
    model_dr = MODELS.resolve_for_use("perplexity/deep-research")
    assert model_dr.identifier == "pplx_alpha"
    assert model_dr.thinking_only is True


def test_unseparated_thinking_suffix_resolution() -> None:
    # claude45haikuthinking resolves to base claude45haiku with thinking identifier
    with warns(ModelRiskWarning):
        model_haiku = MODELS.resolve_for_use("claude45haikuthinking", allow_risky_model=True)
    assert model_haiku.id == "anthropic/claude45haiku"
    assert model_haiku.identifier == "claude45haikuthinking"

    # kimik2thinking resolves to moonshot/kimik2
    with warns(ModelRiskWarning):
        model_k2 = MODELS.resolve_for_use("kimik2thinking", allow_risky_model=True)
    assert model_k2.id == "moonshot/kimik2"
    assert model_k2.identifier == "kimik2thinking"


def test_mid_name_and_effort_thinking_resolution() -> None:
    # gpt51-low-thinking resolves to gpt51 with thinking identifier
    with warns(ModelRiskWarning):
        model_gpt51 = MODELS.resolve_for_use("openai/gpt51-low-thinking", allow_risky_model=True)
    assert model_gpt51.id == "openai/gpt51"
    assert model_gpt51.identifier == "gpt51_thinking"

    # gemini-3.1-pro-thinking-high resolves to gemini-3.1-pro with gemini31pro_high
    model_gemini = MODELS.resolve_for_use("google/gemini-3.1-pro-thinking-high")
    assert model_gemini.id == "google/gemini-3.1-pro"
    assert model_gemini.identifier == "gemini31pro_high"


def test_explicit_non_thinking_modifier_resolution() -> None:
    # grok4nonthinking resolves to grok4 with base identifier
    with warns(ModelRiskWarning):
        model_grok = MODELS.resolve_for_use("x-ai/grok4nonthinking", allow_risky_model=True)
    assert model_grok.id == "x-ai/grok4"
    assert model_grok.identifier == "grok4nonthinking"
